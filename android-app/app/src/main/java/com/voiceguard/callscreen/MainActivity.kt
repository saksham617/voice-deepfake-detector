package com.voiceguard.callscreen

import android.Manifest
import android.app.role.RoleManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.voiceguard.callscreen.databinding.ActivityMainBinding
import rikka.shizuku.Shizuku

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "VoiceGuardMain"
        private const val SHIZUKU_REQUEST_CODE = 1001
    }

    private lateinit var binding: ActivityMainBinding

    private val requestRoleLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            Log.d(TAG, "Role request result: resultCode=${result.resultCode}")
            updateStatus()
        }

    private val requestOverlayLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            Log.d(TAG, "Overlay permission request result: resultCode=${result.resultCode}")
            updateStatus()
        }

    private val requestPhoneStateLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
            Log.d(TAG, "Phone-state permission results: $results")
            updateStatus()
        }

    private val requestNotificationLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            Log.d(TAG, "Notification permission result: granted=$granted")
            updateStatus()
        }

    private val shizukuPermissionListener =
        Shizuku.OnRequestPermissionResultListener { requestCode, grantResult ->
            if (requestCode == SHIZUKU_REQUEST_CODE) {
                Log.d(TAG, "Shizuku permission result: granted=${grantResult == PackageManager.PERMISSION_GRANTED}")
                runOnUiThread { updateStatus() }
            }
        }

    private val shizukuBinderListener = Shizuku.OnBinderReceivedListener {
        Log.d(TAG, "Shizuku binder received (Shizuku is running)")
        runOnUiThread { updateStatus() }
    }

    private val shizukuBinderDeadListener = Shizuku.OnBinderDeadListener {
        Log.d(TAG, "Shizuku binder died (Shizuku stopped running)")
        runOnUiThread { updateStatus() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        Shizuku.addRequestPermissionResultListener(shizukuPermissionListener)
        Shizuku.addBinderReceivedListenerSticky(shizukuBinderListener)
        Shizuku.addBinderDeadListener(shizukuBinderDeadListener)

        binding.requestRoleButton.setOnClickListener {
            requestCallScreeningRole()
        }

        binding.requestOverlayButton.setOnClickListener {
            requestOverlayPermission()
        }

        binding.requestAudioButton.setOnClickListener {
            requestPhoneStatePermission()
        }

        binding.requestShizukuButton.setOnClickListener {
            requestShizukuPermission()
        }

        binding.requestNotificationButton.setOnClickListener {
            requestNotificationPermission()
        }

        updateStatus()
    }

    override fun onDestroy() {
        super.onDestroy()
        Shizuku.removeRequestPermissionResultListener(shizukuPermissionListener)
        Shizuku.removeBinderReceivedListener(shizukuBinderListener)
        Shizuku.removeBinderDeadListener(shizukuBinderDeadListener)
    }

    override fun onResume() {
        super.onResume()
        updateStatus()
    }

    private fun requestCallScreeningRole() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            Log.d(TAG, "Call screening role requires API 29+; device is ${Build.VERSION.SDK_INT}")
            return
        }

        val roleManager = getSystemService(Context.ROLE_SERVICE) as RoleManager

        if (!roleManager.isRoleAvailable(RoleManager.ROLE_CALL_SCREENING)) {
            Log.d(TAG, "ROLE_CALL_SCREENING not available on this device")
            return
        }

        if (roleManager.isRoleHeld(RoleManager.ROLE_CALL_SCREENING)) {
            Log.d(TAG, "ROLE_CALL_SCREENING already held")
            return
        }

        Log.d(TAG, "Launching role request intent for ROLE_CALL_SCREENING")
        val intent = roleManager.createRequestRoleIntent(RoleManager.ROLE_CALL_SCREENING)
        requestRoleLauncher.launch(intent)
    }

    private fun requestOverlayPermission() {
        if (Settings.canDrawOverlays(this)) {
            Log.d(TAG, "Overlay permission already granted")
            return
        }

        Log.d(TAG, "Launching overlay permission settings screen")
        val intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:$packageName")
        )
        requestOverlayLauncher.launch(intent)
    }

    private fun hasPhoneStatePermission(): Boolean {
        val phoneState = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) ==
            PackageManager.PERMISSION_GRANTED
        // RECORD_AUDIO isn't used to capture audio ourselves (Shizuku's helper does that),
        // but it's still required to declare a "microphone" foreground service type at all.
        val recordAudio = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        return phoneState && recordAudio
    }

    private fun requestPhoneStatePermission() {
        if (hasPhoneStatePermission()) {
            Log.d(TAG, "READ_PHONE_STATE and RECORD_AUDIO already granted")
            return
        }

        Log.d(TAG, "Requesting READ_PHONE_STATE and RECORD_AUDIO")
        requestPhoneStateLauncher.launch(
            arrayOf(Manifest.permission.READ_PHONE_STATE, Manifest.permission.RECORD_AUDIO)
        )
    }

    /** Whether the Shizuku app is installed and its service is currently running. */
    private fun isShizukuRunning(): Boolean {
        return try {
            Shizuku.pingBinder()
        } catch (e: Exception) {
            false
        }
    }

    private fun hasShizukuPermission(): Boolean {
        return isShizukuRunning() && Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED
    }

    private fun requestShizukuPermission() {
        if (!isShizukuRunning()) {
            Log.d(TAG, "Shizuku is not running; cannot request permission yet")
            return
        }
        if (hasShizukuPermission()) {
            Log.d(TAG, "Shizuku permission already granted")
            return
        }
        if (Shizuku.shouldShowRequestPermissionRationale()) {
            Log.d(TAG, "User previously denied Shizuku permission")
        }

        Log.d(TAG, "Requesting Shizuku permission")
        Shizuku.requestPermission(SHIZUKU_REQUEST_CODE)
    }

    private fun hasNotificationPermission(): Boolean {
        // POST_NOTIFICATIONS only exists as a runtime permission from API 33 (Tiramisu); below
        // that, notifications just work once the channel is created, no grant needed.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return true
        return ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        if (hasNotificationPermission()) {
            Log.d(TAG, "POST_NOTIFICATIONS already granted")
            return
        }

        Log.d(TAG, "Requesting POST_NOTIFICATIONS")
        requestNotificationLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    private fun updateStatus() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            binding.statusText.text = getString(R.string.status_unsupported)
            return
        }

        val roleManager = getSystemService(Context.ROLE_SERVICE) as RoleManager
        val held = roleManager.isRoleHeld(RoleManager.ROLE_CALL_SCREENING)
        Log.d(TAG, "isRoleHeld(ROLE_CALL_SCREENING) = $held")

        binding.statusText.text = if (held) {
            getString(R.string.status_held)
        } else {
            getString(R.string.status_not_held)
        }
        binding.requestRoleButton.isEnabled = !held

        val overlayGranted = Settings.canDrawOverlays(this)
        Log.d(TAG, "canDrawOverlays = $overlayGranted")

        binding.overlayStatusText.text = if (overlayGranted) {
            getString(R.string.status_overlay_granted)
        } else {
            getString(R.string.status_overlay_not_granted)
        }
        binding.requestOverlayButton.isEnabled = !overlayGranted

        val phoneStateGranted = hasPhoneStatePermission()
        Log.d(TAG, "hasPhoneStatePermission = $phoneStateGranted")

        binding.audioStatusText.text = if (phoneStateGranted) {
            getString(R.string.status_audio_granted)
        } else {
            getString(R.string.status_audio_not_granted)
        }
        binding.requestAudioButton.isEnabled = !phoneStateGranted

        val shizukuRunning = isShizukuRunning()
        val shizukuGranted = hasShizukuPermission()
        Log.d(TAG, "shizukuRunning=$shizukuRunning shizukuGranted=$shizukuGranted")

        binding.shizukuStatusText.text = when {
            shizukuGranted -> getString(R.string.status_shizuku_granted)
            shizukuRunning -> getString(R.string.status_shizuku_not_granted)
            else -> getString(R.string.status_shizuku_not_running)
        }
        binding.requestShizukuButton.isEnabled = shizukuRunning && !shizukuGranted

        val notificationsGranted = hasNotificationPermission()
        Log.d(TAG, "hasNotificationPermission = $notificationsGranted")

        binding.notificationStatusText.text = if (notificationsGranted) {
            getString(R.string.status_notifications_granted)
        } else {
            getString(R.string.status_notifications_not_granted)
        }
        binding.requestNotificationButton.isEnabled = !notificationsGranted
    }
}
