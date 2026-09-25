package com.voiceguard.callscreen

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.Settings
import android.telecom.Call
import android.telecom.CallScreeningService
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * Stage 1: detect that a real call is ringing, allow every call through unchanged.
 * Stage 2: also show a floating overlay with the caller's number for incoming calls.
 * Stage 3: also start the audio-streaming service, which itself waits for the call to
 * be answered before engaging speakerphone/mic/streaming.
 */
class VoiceGuardCallScreeningService : CallScreeningService() {

    companion object {
        private const val TAG = "VoiceGuardScreening"
    }

    override fun onScreenCall(callDetails: Call.Details) {
        val number = callDetails.handle?.schemeSpecificPart ?: "unknown"
        val direction = when (callDetails.callDirection) {
            Call.Details.DIRECTION_INCOMING -> "INCOMING"
            Call.Details.DIRECTION_OUTGOING -> "OUTGOING"
            else -> "UNKNOWN"
        }

        Log.d(TAG, "onScreenCall fired: direction=$direction number=$number")

        if (direction == "INCOMING") {
            showCallerOverlay(number)
            startAudioStreaming(number)
        }

        val response = CallResponse.Builder()
            .setDisallowCall(false)
            .setRejectCall(false)
            .setSkipCallLog(false)
            .setSkipNotification(false)
            .build()

        respondToCall(callDetails, response)
        Log.d(TAG, "respondToCall sent (call allowed through unchanged)")
    }

    private fun showCallerOverlay(number: String) {
        if (!Settings.canDrawOverlays(this)) {
            Log.d(TAG, "Overlay permission not granted; skipping overlay for number=$number")
            return
        }

        val overlayIntent = Intent(this, CallerOverlayService::class.java).apply {
            putExtra(CallerOverlayService.EXTRA_NUMBER, number)
        }
        startService(overlayIntent)
        Log.d(TAG, "Started CallerOverlayService for number=$number")
    }

    private fun startAudioStreaming(number: String) {
        // RECORD_AUDIO is no longer the relevant gate: real call audio is captured by a
        // Shizuku-privileged helper process (shell UID), not by this app's own AudioRecord.
        // CallAudioStreamingService checks Shizuku availability itself once the call is
        // actually answered. READ_PHONE_STATE is still required here for PhoneStateListener.
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(TAG, "READ_PHONE_STATE not granted; skipping audio streaming for number=$number")
            return
        }

        val streamingIntent = Intent(this, CallAudioStreamingService::class.java).apply {
            putExtra(CallAudioStreamingService.EXTRA_NUMBER, number)
        }
        ContextCompat.startForegroundService(this, streamingIntent)
        Log.d(TAG, "Started CallAudioStreamingService for number=$number")
    }
}
