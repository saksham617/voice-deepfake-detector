plugins {
    id("com.android.application")
}

android {
    namespace = "com.voiceguard.callscreen"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.voiceguard.callscreen"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.10.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // Pinned to 11.0.2, not the latest 13.1.5: this version still has Shizuku.newProcess(),
    // which runs a raw command directly (no Zygote fork/specialize step). bindUserService
    // (the only option in 13.1.5) spawns its process through Zygote, which appears to strip
    // Linux capabilities needed for AudioSource.VOICE_CALL even at a matching UID/SELinux
    // domain — confirmed by direct testing, not a guess. newProcess avoids that path entirely.
    implementation("dev.rikka.shizuku:api:11.0.2")
    implementation("dev.rikka.shizuku:provider:11.0.2")
}
