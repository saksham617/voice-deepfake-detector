package com.voiceguard.callscreen

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.os.Build
import android.os.IBinder
import android.telephony.PhoneStateListener
import android.telephony.TelephonyManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import rikka.shizuku.Shizuku
import rikka.shizuku.ShizukuRemoteProcess
import java.util.concurrent.TimeUnit

/**
 * Stage 3 (revived, take 2): once a screened call is actually answered, capture real call
 * audio via a raw process launched by Shizuku.newProcess() — not bindUserService, which
 * spawns its process through Zygote and, confirmed by direct testing, fails to open
 * AudioSource.VOICE_CALL even with a matching UID/SELinux domain to a working raw process.
 * newProcess runs app_process directly (no Zygote fork/specialize step), same as the manual
 * `adb shell app_process` invocation that's been proven to work on this device.
 *
 * A normal in-process AudioRecord on AudioSource.MIC is muted by the OS during a real call —
 * also confirmed by direct testing.
 */
class CallAudioStreamingService : Service() {

    companion object {
        private const val TAG = "VoiceGuardAudio"
        const val EXTRA_NUMBER = "extra_number"

        private const val BACKEND_WS_URL = "ws://192.168.0.190:8000/ws/stream"
        private const val SAMPLE_RATE = 16000
        private const val READ_CHUNK_BYTES = 6400 // 200ms at 16kHz mono 16-bit

        private const val NOTIFICATION_CHANNEL_ID = "voiceguard_call_monitor"
        private const val NOTIFICATION_ID = 42

        private const val ALERT_CHANNEL_ID = "voiceguard_high_risk_alert"
        private const val ALERT_NOTIFICATION_ID = 43
    }

    private lateinit var telephonyManager: TelephonyManager
    private lateinit var audioManager: AudioManager
    private var callerNumber: String = "unknown"

    private var remoteProcess: ShizukuRemoteProcess? = null
    private var readingThread: Thread? = null
    @Volatile private var isStreaming = false

    private var webSocket: WebSocket? = null
    private val httpClient = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var speakerWasOnBeforeCall = false
    @Volatile private var alertNotificationShown = false

    private val phoneStateListener = object : PhoneStateListener() {
        @Deprecated("Deprecated in Java")
        override fun onCallStateChanged(state: Int, phoneNumber: String?) {
            when (state) {
                TelephonyManager.CALL_STATE_OFFHOOK -> {
                    Log.d(TAG, "Call state OFFHOOK (answered) — starting audio pipeline")
                    startAudioPipeline()
                }
                TelephonyManager.CALL_STATE_IDLE -> {
                    Log.d(TAG, "Call state IDLE (ended) — stopping audio pipeline")
                    stopAudioPipelineAndSelf()
                }
                else -> Log.d(TAG, "Call state changed: $state")
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        telephonyManager = getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        callerNumber = intent?.getStringExtra(EXTRA_NUMBER) ?: "unknown"
        Log.d(TAG, "onStartCommand: waiting for call from $callerNumber to connect")

        // Start as a benign foreground-service type first — a cold background start with
        // the microphone type directly is rejected by the OS. Once this service is already
        // running in the foreground, promoting its own type is permitted.
        startForeground(
            NOTIFICATION_ID,
            buildNotification("Waiting for call to connect…"),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        )

        @Suppress("DEPRECATION")
        telephonyManager.listen(phoneStateListener, PhoneStateListener.LISTEN_CALL_STATE)

        return START_NOT_STICKY
    }

    private fun startAudioPipeline() {
        if (isStreaming) {
            Log.d(TAG, "Audio pipeline already running; ignoring duplicate OFFHOOK")
            return
        }

        if (!Shizuku.pingBinder()) {
            Log.d(TAG, "Shizuku is not running; cannot capture real call audio")
            return
        }
        if (Shizuku.checkSelfPermission() != PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Shizuku permission not granted; cannot capture real call audio")
            return
        }

        startForeground(
            NOTIFICATION_ID,
            buildNotification("Monitoring call for deepfake indicators…"),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        )
        alertNotificationShown = false
        engageSpeakerphone()
        connectWebSocket()
        beginCapture()
    }

    private fun beginCapture() {
        // Our own installed APK's path doubles as the classpath app_process needs to find
        // VoiceCallCaptureMain — it's compiled into this app normally by Gradle, no separate
        // manual dex step required.
        val apkPath = applicationInfo.sourceDir
        // Passing CLASSPATH via newProcess's `environment` array likely REPLACES the whole
        // process environment rather than adding to it, stripping variables app_process/ART
        // actually needs to boot (ANDROID_ROOT, BOOTCLASSPATH, etc.) — confirmed the identical
        // command works fine when run through an interactive shell (full inherited env) but
        // fails silently (no stdout, no stderr) through newProcess with a sparse custom env.
        // Wrapping in `sh -c` sets CLASSPATH via shell syntax while inheriting environment
        // normally, matching exactly how the working manual test actually ran under the hood.
        val cmd = arrayOf(
            "/system/bin/sh",
            "-c",
            "CLASSPATH=$apkPath exec app_process / com.voiceguard.callscreen.VoiceCallCaptureMain"
        )

        val process = try {
            Shizuku.newProcess(cmd, null, "/")
        } catch (e: Exception) {
            Log.e(TAG, "Shizuku.newProcess failed", e)
            null
        }

        if (process == null) {
            Log.e(TAG, "newProcess returned null; real audio capture unavailable")
            return
        }

        remoteProcess = process
        isStreaming = true
        Log.d(TAG, "Privileged process launched, reading from its stdout")

        readingThread = Thread {
            val input = process.inputStream
            val errInput = process.errorStream
            val chunk = ByteArray(READ_CHUNK_BYTES)
            var chunkCount = 0
            try {
                while (isStreaming) {
                    val bytesRead = input.read(chunk)
                    if (bytesRead > 0) {
                        if (chunkCount % 5 == 0) {
                            Log.d(TAG, "Call audio chunk peak amplitude: ${peakAmplitude(chunk, bytesRead)} / 32767")
                        }
                        chunkCount++
                        webSocket?.send(chunk.copyOf(bytesRead).toByteString())
                    } else if (bytesRead < 0) {
                        Log.d(TAG, "Privileged process stdout closed (read returned $bytesRead)")
                        val stderrText = errInput.bufferedReader().readText()
                        if (stderrText.isNotBlank()) {
                            Log.e(TAG, "Privileged process stderr: $stderrText")
                        }
                        break
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Reading from privileged process failed", e)
            }
        }.also {
            it.name = "VoiceCallAudioReader"
            it.start()
        }
    }

    private fun engageSpeakerphone() {
        speakerWasOnBeforeCall = audioManager.isSpeakerphoneOn
        audioManager.isSpeakerphoneOn = true
        Log.d(TAG, "Speakerphone engaged (was $speakerWasOnBeforeCall before)")
    }

    private fun connectWebSocket() {
        Log.d(TAG, "Connecting to backend: $BACKEND_WS_URL")
        val request = Request.Builder().url(BACKEND_WS_URL).build()
        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "WebSocket open; sending sample-rate config")
                val config = JSONObject().apply {
                    put("type", "config")
                    put("input_sample_rate", SAMPLE_RATE)
                }
                webSocket.send(config.toString())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleServerMessage(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure: ${t.message}", t)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket closing: code=$code reason=$reason")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket closed: code=$code reason=$reason")
            }
        })
    }

    private fun handleServerMessage(text: String) {
        val json = try {
            JSONObject(text)
        } catch (e: Exception) {
            Log.e(TAG, "Could not parse server message: $text", e)
            return
        }

        when (json.optString("type")) {
            "ready" -> Log.d(TAG, "Backend session ready: session_id=${json.optString("session_id")}")
            "score" -> {
                val fakeProb = json.optDouble("fake_prob", -1.0)
                val risk = json.optJSONObject("risk")
                val level = risk?.optString("level") ?: "unknown"
                val alert = json.optBoolean("alert", false)
                Log.d(
                    TAG,
                    "SCORE index=${json.optInt("index")} fake_prob=$fakeProb risk_level=$level alert=$alert"
                )
                CallerOverlayService.updateRiskStatus(fakeProb, level, alert)
                if (alert && !alertNotificationShown) {
                    alertNotificationShown = true
                    postAlertNotification(fakeProb)
                }
            }
            else -> Log.d(TAG, "Server message: $text")
        }
    }

    private fun peakAmplitude(bytes: ByteArray, length: Int): Int {
        var peak = 0
        var i = 0
        while (i + 1 < length) {
            val sample = ((bytes[i + 1].toInt() shl 8) or (bytes[i].toInt() and 0xFF)).toShort().toInt()
            val abs = kotlin.math.abs(sample)
            if (abs > peak) peak = abs
            i += 2
        }
        return peak
    }

    private fun stopAudioPipelineAndSelf() {
        isStreaming = false
        readingThread?.join(500)
        readingThread = null

        try {
            remoteProcess?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Destroying privileged process failed", e)
        }
        remoteProcess = null

        webSocket?.send(JSONObject().apply { put("type", "end") }.toString())
        webSocket?.close(1000, "call ended")
        webSocket = null

        audioManager.isSpeakerphoneOn = speakerWasOnBeforeCall
        Log.d(TAG, "Speakerphone restored to $speakerWasOnBeforeCall")

        @Suppress("DEPRECATION")
        telephonyManager.listen(phoneStateListener, PhoneStateListener.LISTEN_NONE)

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isStreaming) {
            stopAudioPipelineAndSelf()
        }
    }

    private fun createNotificationChannel() {
        val notificationManager = getSystemService(NotificationManager::class.java)

        val monitorChannel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            "Call Monitoring",
            NotificationManager.IMPORTANCE_LOW
        )
        notificationManager.createNotificationChannel(monitorChannel)

        // Separate, high-importance channel so a HIGH-risk alert gets sound + vibration —
        // deliberately not just a louder version of the silent "monitoring" notification,
        // since the whole point is to be noticeable even if the user isn't looking at the
        // screen mid-conversation.
        val alertChannel = NotificationChannel(
            ALERT_CHANNEL_ID,
            "Deepfake Risk Alerts",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            enableVibration(true)
            enableLights(true)
        }
        notificationManager.createNotificationChannel(alertChannel)
    }

    private fun buildNotification(text: String): Notification {
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("VoiceGuard")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun postAlertNotification(fakeProb: Double) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(TAG, "POST_NOTIFICATIONS not granted; skipping alert notification")
            return
        }

        val percent = (fakeProb * 100).toInt()
        val notification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setContentTitle("⚠ VoiceGuard: High deepfake risk")
            .setContentText("Call with $callerNumber is showing sustained signs of a synthetic voice ($percent%).")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setAutoCancel(true)
            .build()

        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.notify(ALERT_NOTIFICATION_ID, notification)
        Log.d(TAG, "Posted HIGH-risk alert notification ($percent%)")
    }

}
