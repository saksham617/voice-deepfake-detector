package com.voiceguard.callscreen

import android.app.Service
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.ContextThemeWrapper
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.TextView

/**
 * Stage 2: a floating overlay showing the caller's number while a call is ringing.
 * Also shows the live risk score once Stage 3's audio pipeline starts producing results —
 * CallAudioStreamingService pushes updates via the companion object's updateRiskStatus(),
 * since both run as plain Services in the same app process.
 */
class CallerOverlayService : Service() {

    companion object {
        private const val TAG = "VoiceGuardOverlay"
        const val EXTRA_NUMBER = "extra_number"

        private val mainHandler = Handler(Looper.getMainLooper())
        @Volatile private var activeInstance: CallerOverlayService? = null

        /** Safe to call from any thread (e.g. the WebSocket callback thread); no-ops if no overlay is currently showing. */
        fun updateRiskStatus(fakeProb: Double, level: String, alert: Boolean) {
            val instance = activeInstance ?: return
            mainHandler.post { instance.applyRiskStatus(fakeProb, level, alert) }
        }
    }

    private var windowManager: WindowManager? = null
    private var overlayView: View? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        activeInstance = this
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val number = intent?.getStringExtra(EXTRA_NUMBER) ?: "unknown"
        Log.d(TAG, "onStartCommand: showing overlay for number=$number")
        showOverlay(number)
        return START_NOT_STICKY
    }

    private fun showOverlay(number: String) {
        val existingView = overlayView
        if (existingView != null) {
            existingView.findViewById<TextView>(R.id.overlayNumberText).text = number
            existingView.findViewById<TextView>(R.id.overlayRiskText).text =
                getString(R.string.overlay_risk_waiting)
            existingView.findViewById<View>(R.id.overlayRoot)
                .setBackgroundResource(R.drawable.overlay_background)
            return
        }

        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        windowManager = wm

        val themedContext = ContextThemeWrapper(this, R.style.Theme_VoiceGuardCallScreen)
        val view = LayoutInflater.from(themedContext).inflate(R.layout.overlay_caller_info, null)
        view.findViewById<TextView>(R.id.overlayNumberText).text = number
        view.findViewById<View>(R.id.overlayDismissButton).setOnClickListener {
            Log.d(TAG, "Overlay dismissed by user")
            stopSelf()
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        )
        params.gravity = Gravity.TOP
        params.y = 100

        wm.addView(view, params)
        overlayView = view
        Log.d(TAG, "Overlay view added to window")
    }

    private fun applyRiskStatus(fakeProb: Double, level: String, alert: Boolean) {
        val view = overlayView ?: return
        val riskText = view.findViewById<TextView>(R.id.overlayRiskText)
        val percent = (fakeProb * 100).toInt()

        // Only two tiers are shown to the user: HIGH (backend's sustained-high verdict)
        // and LOW (everything else — the backend's NONE/LOW/MEDIUM are all "not yet HIGH").
        val isHigh = alert || level == "HIGH"
        val displayLevel = if (isHigh) "HIGH" else "LOW"

        riskText.text = if (isHigh) {
            getString(R.string.overlay_risk_alert, percent)
        } else {
            getString(R.string.overlay_risk_format, displayLevel, percent)
        }

        val root = view.findViewById<View>(R.id.overlayRoot)
        val backgroundRes = if (isHigh) R.drawable.overlay_background_high else R.drawable.overlay_background
        root.setBackgroundResource(backgroundRes)
        Log.d(TAG, "applyRiskStatus: backendLevel=$level alert=$alert -> displayLevel=$displayLevel background=${resourceLabel(backgroundRes)}")
    }

    private fun resourceLabel(resId: Int): String = when (resId) {
        R.drawable.overlay_background_high -> "RED (high)"
        R.drawable.overlay_background -> "GREEN (low)"
        else -> resId.toString()
    }

    override fun onDestroy() {
        super.onDestroy()
        overlayView?.let {
            windowManager?.removeView(it)
            Log.d(TAG, "Overlay view removed")
        }
        overlayView = null
        windowManager = null
        if (activeInstance === this) {
            activeInstance = null
        }
    }
}
