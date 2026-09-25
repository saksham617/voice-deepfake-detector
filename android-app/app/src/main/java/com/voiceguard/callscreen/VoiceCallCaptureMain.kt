package com.voiceguard.callscreen

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

/**
 * Entry point for a raw process launched via Shizuku.newProcess() (app_process, invoked
 * directly by Shizuku's server — no Zygote fork/specialize step). Confirmed by direct,
 * repeated testing that Zygote-forked processes (Shizuku's bindUserService, the only other
 * option) fail to open AudioSource.VOICE_CALL even with a matching UID and SELinux domain to
 * a working raw process — Zygote specialization evidently strips something (likely Linux
 * capabilities) that a directly-exec'd shell process keeps. This class runs in that directly-
 * exec'd process instead, as plain main(), not a normal Android app component.
 *
 * Writes raw 16kHz mono PCM16 continuously to stdout; the main app reads it via
 * ShizukuRemoteProcess's InputStream. Terminated externally via ShizukuRemoteProcess.destroy().
 */
object VoiceCallCaptureMain {

    private const val SAMPLE_RATE = 16000
    private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

    @JvmStatic
    fun main(args: Array<String>) {
        val minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (minBufferSize <= 0) {
            System.err.println("getMinBufferSize failed: $minBufferSize")
            return
        }

        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_CALL,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                maxOf(minBufferSize, 6400)
            )
        } catch (e: Exception) {
            System.err.println("AudioRecord construction failed: $e")
            return
        }

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            System.err.println("AudioRecord not initialized, state=${record.state}")
            record.release()
            return
        }

        record.startRecording()
        System.err.println("Recording started, uid=${android.os.Process.myUid()}")

        val out = System.out
        val buf = ByteArray(6400)
        try {
            while (true) {
                val n = record.read(buf, 0, buf.size)
                if (n > 0) {
                    out.write(buf, 0, n)
                    out.flush()
                }
            }
        } catch (e: Exception) {
            System.err.println("capture loop ended: $e")
        } finally {
            record.stop()
            record.release()
        }
    }
}
