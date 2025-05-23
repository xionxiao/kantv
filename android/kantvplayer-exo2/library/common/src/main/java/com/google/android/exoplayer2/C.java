/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.android.exoplayer2;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.MediaCodec;
import android.media.MediaFormat;
import androidx.annotation.IntDef;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import com.google.android.exoplayer2.util.MimeTypes;
import com.google.android.exoplayer2.util.Util;
import java.lang.annotation.Documented;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.UUID;

/** Defines constants used by the library. */
@SuppressWarnings("InlinedApi")
public final class C {

  private C() {}

  /**
   * Special constant representing a time corresponding to the end of a source. Suitable for use in
   * any time base.
   */
  public static final long TIME_END_OF_SOURCE = Long.MIN_VALUE;

  /**
   * Special constant representing an unset or unknown time or duration. Suitable for use in any
   * time base.
   */
  public static final long TIME_UNSET = Long.MIN_VALUE + 1;

  /** Represents an unset or unknown index. */
  public static final int INDEX_UNSET = -1;

  /** Represents an unset or unknown position. */
  public static final int POSITION_UNSET = -1;

  /** Represents an unset or unknown rate. */
  public static final float RATE_UNSET = -Float.MAX_VALUE;

  /** Represents an unset or unknown length. */
  public static final int LENGTH_UNSET = -1;

  /** Represents an unset or unknown percentage. */
  public static final int PERCENTAGE_UNSET = -1;

  /** The number of milliseconds in one second. */
  public static final long MILLIS_PER_SECOND = 1000L;

  /** The number of microseconds in one second. */
  public static final long MICROS_PER_SECOND = 1000000L;

  /** The number of nanoseconds in one second. */
  public static final long NANOS_PER_SECOND = 1000000000L;

  /** The number of bits per byte. */
  public static final int BITS_PER_BYTE = 8;

  /** The number of bytes per float. */
  public static final int BYTES_PER_FLOAT = 4;

  /**
   * @deprecated Use {@link java.nio.charset.StandardCharsets} or {@link
   *     com.google.common.base.Charsets} instead.
   */
  @Deprecated public static final String ASCII_NAME = "US-ASCII";

  /**
   * @deprecated Use {@link java.nio.charset.StandardCharsets} or {@link
   *     com.google.common.base.Charsets} instead.
   */
  @Deprecated public static final String UTF8_NAME = "UTF-8";

  /**
   * @deprecated Use {@link java.nio.charset.StandardCharsets} or {@link
   *     com.google.common.base.Charsets} instead.
   */
  @Deprecated public static final String ISO88591_NAME = "ISO-8859-1";

  /**
   * @deprecated Use {@link java.nio.charset.StandardCharsets} or {@link
   *     com.google.common.base.Charsets} instead.
   */
  @Deprecated public static final String UTF16_NAME = "UTF-16";

  /**
   * @deprecated Use {@link java.nio.charset.StandardCharsets} or {@link
   *     com.google.common.base.Charsets} instead.
   */
  @Deprecated public static final String UTF16LE_NAME = "UTF-16LE";

  /** The name of the serif font family. */
  public static final String SERIF_NAME = "serif";

  /** The name of the sans-serif font family. */
  public static final String SANS_SERIF_NAME = "sans-serif";

  /**
   * Crypto modes for a codec. One of {@link #CRYPTO_MODE_UNENCRYPTED}, {@link #CRYPTO_MODE_AES_CTR}
   * or {@link #CRYPTO_MODE_AES_CBC}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({CRYPTO_MODE_UNENCRYPTED, CRYPTO_MODE_AES_CTR, CRYPTO_MODE_AES_CBC})
  public @interface CryptoMode {}
  /** @see MediaCodec#CRYPTO_MODE_UNENCRYPTED */
  public static final int CRYPTO_MODE_UNENCRYPTED = MediaCodec.CRYPTO_MODE_UNENCRYPTED;
  /** @see MediaCodec#CRYPTO_MODE_AES_CTR */
  public static final int CRYPTO_MODE_AES_CTR = MediaCodec.CRYPTO_MODE_AES_CTR;
  /** @see MediaCodec#CRYPTO_MODE_AES_CBC */
  public static final int CRYPTO_MODE_AES_CBC = MediaCodec.CRYPTO_MODE_AES_CBC;

  //weiguo: make native multidrm & wiseplay happy
  public static final int CRYPTO_MODE_SM4_CTR     = 3;
  public static final int CRYPTO_MODE_SM4_CBC     = 4;

  /**
   * Represents an unset {@link android.media.AudioTrack} session identifier. Equal to {@link
   * AudioManager#AUDIO_SESSION_ID_GENERATE}.
   */
  public static final int AUDIO_SESSION_ID_UNSET = AudioManager.AUDIO_SESSION_ID_GENERATE;

  /**
   * Represents an audio encoding, or an invalid or unset value. One of {@link Format#NO_VALUE},
   * {@link #ENCODING_INVALID}, {@link #ENCODING_PCM_8BIT}, {@link #ENCODING_PCM_16BIT}, {@link
   * #ENCODING_PCM_16BIT_BIG_ENDIAN}, {@link #ENCODING_PCM_24BIT}, {@link #ENCODING_PCM_32BIT},
   * {@link #ENCODING_PCM_FLOAT}, {@link #ENCODING_MP3}, {@link #ENCODING_AC3}, {@link
   * #ENCODING_E_AC3}, {@link #ENCODING_E_AC3_JOC}, {@link #ENCODING_AC4}, {@link #ENCODING_DTS},
   * {@link #ENCODING_DTS_HD} or {@link #ENCODING_DOLBY_TRUEHD}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    Format.NO_VALUE,
    ENCODING_INVALID,
    ENCODING_PCM_8BIT,
    ENCODING_PCM_16BIT,
    ENCODING_PCM_16BIT_BIG_ENDIAN,
    ENCODING_PCM_24BIT,
    ENCODING_PCM_32BIT,
    ENCODING_PCM_FLOAT,
    ENCODING_MP3,
    ENCODING_AAC_LC,
    ENCODING_AAC_HE_V1,
    ENCODING_AAC_HE_V2,
    ENCODING_AAC_XHE,
    ENCODING_AAC_ELD,
    ENCODING_AAC_ER_BSAC,
    ENCODING_AC3,
    ENCODING_E_AC3,
    ENCODING_E_AC3_JOC,
    ENCODING_AC4,
    ENCODING_DTS,
    ENCODING_DTS_HD,
    ENCODING_DOLBY_TRUEHD
  })
  public @interface Encoding {}

  /**
   * Represents a PCM audio encoding, or an invalid or unset value. One of {@link Format#NO_VALUE},
   * {@link #ENCODING_INVALID}, {@link #ENCODING_PCM_8BIT}, {@link #ENCODING_PCM_16BIT}, {@link
   * #ENCODING_PCM_16BIT_BIG_ENDIAN}, {@link #ENCODING_PCM_24BIT}, {@link #ENCODING_PCM_32BIT},
   * {@link #ENCODING_PCM_FLOAT}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    Format.NO_VALUE,
    ENCODING_INVALID,
    ENCODING_PCM_8BIT,
    ENCODING_PCM_16BIT,
    ENCODING_PCM_16BIT_BIG_ENDIAN,
    ENCODING_PCM_24BIT,
    ENCODING_PCM_32BIT,
    ENCODING_PCM_FLOAT
  })
  public @interface PcmEncoding {}
  /** @see AudioFormat#ENCODING_INVALID */
  public static final int ENCODING_INVALID = AudioFormat.ENCODING_INVALID;
  /** @see AudioFormat#ENCODING_PCM_8BIT */
  public static final int ENCODING_PCM_8BIT = AudioFormat.ENCODING_PCM_8BIT;
  /** @see AudioFormat#ENCODING_PCM_16BIT */
  public static final int ENCODING_PCM_16BIT = AudioFormat.ENCODING_PCM_16BIT;
  /** Like {@link #ENCODING_PCM_16BIT}, but with the bytes in big endian order. */
  public static final int ENCODING_PCM_16BIT_BIG_ENDIAN = 0x10000000;
  /** PCM encoding with 24 bits per sample. */
  public static final int ENCODING_PCM_24BIT = 0x20000000;
  /** PCM encoding with 32 bits per sample. */
  public static final int ENCODING_PCM_32BIT = 0x30000000;
  /** @see AudioFormat#ENCODING_PCM_FLOAT */
  public static final int ENCODING_PCM_FLOAT = AudioFormat.ENCODING_PCM_FLOAT;
  /** @see AudioFormat#ENCODING_MP3 */
  public static final int ENCODING_MP3 = AudioFormat.ENCODING_MP3;
  /** @see AudioFormat#ENCODING_AAC_LC */
  public static final int ENCODING_AAC_LC = AudioFormat.ENCODING_AAC_LC;
  /** @see AudioFormat#ENCODING_AAC_HE_V1 */
  public static final int ENCODING_AAC_HE_V1 = AudioFormat.ENCODING_AAC_HE_V1;
  /** @see AudioFormat#ENCODING_AAC_HE_V2 */
  public static final int ENCODING_AAC_HE_V2 = AudioFormat.ENCODING_AAC_HE_V2;
  /** @see AudioFormat#ENCODING_AAC_XHE */
  public static final int ENCODING_AAC_XHE = AudioFormat.ENCODING_AAC_XHE;
  /** @see AudioFormat#ENCODING_AAC_ELD */
  public static final int ENCODING_AAC_ELD = AudioFormat.ENCODING_AAC_ELD;
  /** AAC Error Resilient Bit-Sliced Arithmetic Coding. */
  public static final int ENCODING_AAC_ER_BSAC = 0x40000000;
  /** @see AudioFormat#ENCODING_AC3 */
  public static final int ENCODING_AC3 = AudioFormat.ENCODING_AC3;
  /** @see AudioFormat#ENCODING_E_AC3 */
  public static final int ENCODING_E_AC3 = AudioFormat.ENCODING_E_AC3;
  /** @see AudioFormat#ENCODING_E_AC3_JOC */
  public static final int ENCODING_E_AC3_JOC = AudioFormat.ENCODING_E_AC3_JOC;
  /** @see AudioFormat#ENCODING_AC4 */
  public static final int ENCODING_AC4 = AudioFormat.ENCODING_AC4;
  /** @see AudioFormat#ENCODING_DTS */
  public static final int ENCODING_DTS = AudioFormat.ENCODING_DTS;
  /** @see AudioFormat#ENCODING_DTS_HD */
  public static final int ENCODING_DTS_HD = AudioFormat.ENCODING_DTS_HD;
  /** @see AudioFormat#ENCODING_DOLBY_TRUEHD */
  public static final int ENCODING_DOLBY_TRUEHD = AudioFormat.ENCODING_DOLBY_TRUEHD;

  /**
   * Stream types for an {@link android.media.AudioTrack}. One of {@link #STREAM_TYPE_ALARM}, {@link
   * #STREAM_TYPE_DTMF}, {@link #STREAM_TYPE_MUSIC}, {@link #STREAM_TYPE_NOTIFICATION}, {@link
   * #STREAM_TYPE_RING}, {@link #STREAM_TYPE_SYSTEM} or {@link #STREAM_TYPE_VOICE_CALL}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    STREAM_TYPE_ALARM,
    STREAM_TYPE_DTMF,
    STREAM_TYPE_MUSIC,
    STREAM_TYPE_NOTIFICATION,
    STREAM_TYPE_RING,
    STREAM_TYPE_SYSTEM,
    STREAM_TYPE_VOICE_CALL
  })
  public @interface StreamType {}
  /** @see AudioManager#STREAM_ALARM */
  public static final int STREAM_TYPE_ALARM = AudioManager.STREAM_ALARM;
  /** @see AudioManager#STREAM_DTMF */
  public static final int STREAM_TYPE_DTMF = AudioManager.STREAM_DTMF;
  /** @see AudioManager#STREAM_MUSIC */
  public static final int STREAM_TYPE_MUSIC = AudioManager.STREAM_MUSIC;
  /** @see AudioManager#STREAM_NOTIFICATION */
  public static final int STREAM_TYPE_NOTIFICATION = AudioManager.STREAM_NOTIFICATION;
  /** @see AudioManager#STREAM_RING */
  public static final int STREAM_TYPE_RING = AudioManager.STREAM_RING;
  /** @see AudioManager#STREAM_SYSTEM */
  public static final int STREAM_TYPE_SYSTEM = AudioManager.STREAM_SYSTEM;
  /** @see AudioManager#STREAM_VOICE_CALL */
  public static final int STREAM_TYPE_VOICE_CALL = AudioManager.STREAM_VOICE_CALL;
  /** The default stream type used by audio renderers. Equal to {@link #STREAM_TYPE_MUSIC}. */
  public static final int STREAM_TYPE_DEFAULT = STREAM_TYPE_MUSIC;

  /**
   * Content types for audio attributes. One of {@link #CONTENT_TYPE_MOVIE}, {@link
   * #CONTENT_TYPE_MUSIC}, {@link #CONTENT_TYPE_SONIFICATION}, {@link #CONTENT_TYPE_SPEECH} or
   * {@link #CONTENT_TYPE_UNKNOWN}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    CONTENT_TYPE_MOVIE,
    CONTENT_TYPE_MUSIC,
    CONTENT_TYPE_SONIFICATION,
    CONTENT_TYPE_SPEECH,
    CONTENT_TYPE_UNKNOWN
  })
  public @interface AudioContentType {}
  /** @see android.media.AudioAttributes#CONTENT_TYPE_MOVIE */
  public static final int CONTENT_TYPE_MOVIE = android.media.AudioAttributes.CONTENT_TYPE_MOVIE;
  /** @see android.media.AudioAttributes#CONTENT_TYPE_MUSIC */
  public static final int CONTENT_TYPE_MUSIC = android.media.AudioAttributes.CONTENT_TYPE_MUSIC;
  /** @see android.media.AudioAttributes#CONTENT_TYPE_SONIFICATION */
  public static final int CONTENT_TYPE_SONIFICATION =
      android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION;
  /** @see android.media.AudioAttributes#CONTENT_TYPE_SPEECH */
  public static final int CONTENT_TYPE_SPEECH = android.media.AudioAttributes.CONTENT_TYPE_SPEECH;
  /** @see android.media.AudioAttributes#CONTENT_TYPE_UNKNOWN */
  public static final int CONTENT_TYPE_UNKNOWN = android.media.AudioAttributes.CONTENT_TYPE_UNKNOWN;

  /**
   * Flags for audio attributes. Possible flag value is {@link #FLAG_AUDIBILITY_ENFORCED}.
   *
   * <p>Note that {@code FLAG_HW_AV_SYNC} is not available because the player takes care of setting
   * the flag when tunneling is enabled via a track selector.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef(
      flag = true,
      value = {FLAG_AUDIBILITY_ENFORCED})
  public @interface AudioFlags {}
  /** @see android.media.AudioAttributes#FLAG_AUDIBILITY_ENFORCED */
  public static final int FLAG_AUDIBILITY_ENFORCED =
      android.media.AudioAttributes.FLAG_AUDIBILITY_ENFORCED;

  /**
   * Usage types for audio attributes. One of {@link #USAGE_ALARM}, {@link
   * #USAGE_ASSISTANCE_ACCESSIBILITY}, {@link #USAGE_ASSISTANCE_NAVIGATION_GUIDANCE}, {@link
   * #USAGE_ASSISTANCE_SONIFICATION}, {@link #USAGE_ASSISTANT}, {@link #USAGE_GAME}, {@link
   * #USAGE_MEDIA}, {@link #USAGE_NOTIFICATION}, {@link #USAGE_NOTIFICATION_COMMUNICATION_DELAYED},
   * {@link #USAGE_NOTIFICATION_COMMUNICATION_INSTANT}, {@link
   * #USAGE_NOTIFICATION_COMMUNICATION_REQUEST}, {@link #USAGE_NOTIFICATION_EVENT}, {@link
   * #USAGE_NOTIFICATION_RINGTONE}, {@link #USAGE_UNKNOWN}, {@link #USAGE_VOICE_COMMUNICATION} or
   * {@link #USAGE_VOICE_COMMUNICATION_SIGNALLING}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    USAGE_ALARM,
    USAGE_ASSISTANCE_ACCESSIBILITY,
    USAGE_ASSISTANCE_NAVIGATION_GUIDANCE,
    USAGE_ASSISTANCE_SONIFICATION,
    USAGE_ASSISTANT,
    USAGE_GAME,
    USAGE_MEDIA,
    USAGE_NOTIFICATION,
    USAGE_NOTIFICATION_COMMUNICATION_DELAYED,
    USAGE_NOTIFICATION_COMMUNICATION_INSTANT,
    USAGE_NOTIFICATION_COMMUNICATION_REQUEST,
    USAGE_NOTIFICATION_EVENT,
    USAGE_NOTIFICATION_RINGTONE,
    USAGE_UNKNOWN,
    USAGE_VOICE_COMMUNICATION,
    USAGE_VOICE_COMMUNICATION_SIGNALLING
  })
  public @interface AudioUsage {}
  /** @see android.media.AudioAttributes#USAGE_ALARM */
  public static final int USAGE_ALARM = android.media.AudioAttributes.USAGE_ALARM;
  /** @see android.media.AudioAttributes#USAGE_ASSISTANCE_ACCESSIBILITY */
  public static final int USAGE_ASSISTANCE_ACCESSIBILITY =
      android.media.AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY;
  /** @see android.media.AudioAttributes#USAGE_ASSISTANCE_NAVIGATION_GUIDANCE */
  public static final int USAGE_ASSISTANCE_NAVIGATION_GUIDANCE =
      android.media.AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE;
  /** @see android.media.AudioAttributes#USAGE_ASSISTANCE_SONIFICATION */
  public static final int USAGE_ASSISTANCE_SONIFICATION =
      android.media.AudioAttributes.USAGE_ASSISTANCE_SONIFICATION;
  /** @see android.media.AudioAttributes#USAGE_ASSISTANT */
  public static final int USAGE_ASSISTANT = android.media.AudioAttributes.USAGE_ASSISTANT;
  /** @see android.media.AudioAttributes#USAGE_GAME */
  public static final int USAGE_GAME = android.media.AudioAttributes.USAGE_GAME;
  /** @see android.media.AudioAttributes#USAGE_MEDIA */
  public static final int USAGE_MEDIA = android.media.AudioAttributes.USAGE_MEDIA;
  /** @see android.media.AudioAttributes#USAGE_NOTIFICATION */
  public static final int USAGE_NOTIFICATION = android.media.AudioAttributes.USAGE_NOTIFICATION;
  /** @see android.media.AudioAttributes#USAGE_NOTIFICATION_COMMUNICATION_DELAYED */
  public static final int USAGE_NOTIFICATION_COMMUNICATION_DELAYED =
      android.media.AudioAttributes.USAGE_NOTIFICATION_COMMUNICATION_DELAYED;
  /** @see android.media.AudioAttributes#USAGE_NOTIFICATION_COMMUNICATION_INSTANT */
  public static final int USAGE_NOTIFICATION_COMMUNICATION_INSTANT =
      android.media.AudioAttributes.USAGE_NOTIFICATION_COMMUNICATION_INSTANT;
  /** @see android.media.AudioAttributes#USAGE_NOTIFICATION_COMMUNICATION_REQUEST */
  public static final int USAGE_NOTIFICATION_COMMUNICATION_REQUEST =
      android.media.AudioAttributes.USAGE_NOTIFICATION_COMMUNICATION_REQUEST;
  /** @see android.media.AudioAttributes#USAGE_NOTIFICATION_EVENT */
  public static final int USAGE_NOTIFICATION_EVENT =
      android.media.AudioAttributes.USAGE_NOTIFICATION_EVENT;
  /** @see android.media.AudioAttributes#USAGE_NOTIFICATION_RINGTONE */
  public static final int USAGE_NOTIFICATION_RINGTONE =
      android.media.AudioAttributes.USAGE_NOTIFICATION_RINGTONE;
  /** @see android.media.AudioAttributes#USAGE_UNKNOWN */
  public static final int USAGE_UNKNOWN = android.media.AudioAttributes.USAGE_UNKNOWN;
  /** @see android.media.AudioAttributes#USAGE_VOICE_COMMUNICATION */
  public static final int USAGE_VOICE_COMMUNICATION =
      android.media.AudioAttributes.USAGE_VOICE_COMMUNICATION;
  /** @see android.media.AudioAttributes#USAGE_VOICE_COMMUNICATION_SIGNALLING */
  public static final int USAGE_VOICE_COMMUNICATION_SIGNALLING =
      android.media.AudioAttributes.USAGE_VOICE_COMMUNICATION_SIGNALLING;

  /**
   * Capture policies for audio attributes. One of {@link #ALLOW_CAPTURE_BY_ALL}, {@link
   * #ALLOW_CAPTURE_BY_NONE} or {@link #ALLOW_CAPTURE_BY_SYSTEM}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({ALLOW_CAPTURE_BY_ALL, ALLOW_CAPTURE_BY_NONE, ALLOW_CAPTURE_BY_SYSTEM})
  public @interface AudioAllowedCapturePolicy {}
  /** See {@link android.media.AudioAttributes#ALLOW_CAPTURE_BY_ALL}. */
  public static final int ALLOW_CAPTURE_BY_ALL = AudioAttributes.ALLOW_CAPTURE_BY_ALL;
  /** See {@link android.media.AudioAttributes#ALLOW_CAPTURE_BY_NONE}. */
  public static final int ALLOW_CAPTURE_BY_NONE = AudioAttributes.ALLOW_CAPTURE_BY_NONE;
  /** See {@link android.media.AudioAttributes#ALLOW_CAPTURE_BY_SYSTEM}. */
  public static final int ALLOW_CAPTURE_BY_SYSTEM = AudioAttributes.ALLOW_CAPTURE_BY_SYSTEM;

  /**
   * Audio focus types. One of {@link #AUDIOFOCUS_NONE}, {@link #AUDIOFOCUS_GAIN}, {@link
   * #AUDIOFOCUS_GAIN_TRANSIENT}, {@link #AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK} or {@link
   * #AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    AUDIOFOCUS_NONE,
    AUDIOFOCUS_GAIN,
    AUDIOFOCUS_GAIN_TRANSIENT,
    AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK,
    AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE
  })
  public @interface AudioFocusGain {}
  /** @see AudioManager#AUDIOFOCUS_NONE */
  public static final int AUDIOFOCUS_NONE = AudioManager.AUDIOFOCUS_NONE;
  /** @see AudioManager#AUDIOFOCUS_GAIN */
  public static final int AUDIOFOCUS_GAIN = AudioManager.AUDIOFOCUS_GAIN;
  /** @see AudioManager#AUDIOFOCUS_GAIN_TRANSIENT */
  public static final int AUDIOFOCUS_GAIN_TRANSIENT = AudioManager.AUDIOFOCUS_GAIN_TRANSIENT;
  /** @see AudioManager#AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK */
  public static final int AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK =
      AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK;
  /** @see AudioManager#AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE */
  public static final int AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE =
      AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE;

  /**
   * Flags which can apply to a buffer containing a media sample. Possible flag values are {@link
   * #BUFFER_FLAG_KEY_FRAME}, {@link #BUFFER_FLAG_END_OF_STREAM}, {@link #BUFFER_FLAG_LAST_SAMPLE},
   * {@link #BUFFER_FLAG_ENCRYPTED} and {@link #BUFFER_FLAG_DECODE_ONLY}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef(
      flag = true,
      value = {
        BUFFER_FLAG_KEY_FRAME,
        BUFFER_FLAG_END_OF_STREAM,
        BUFFER_FLAG_HAS_SUPPLEMENTAL_DATA,
        BUFFER_FLAG_LAST_SAMPLE,
        BUFFER_FLAG_ENCRYPTED,
        BUFFER_FLAG_DECODE_ONLY
      })
  public @interface BufferFlags {}
  /** Indicates that a buffer holds a synchronization sample. */
  public static final int BUFFER_FLAG_KEY_FRAME = MediaCodec.BUFFER_FLAG_KEY_FRAME;
  /** Flag for empty buffers that signal that the end of the stream was reached. */
  public static final int BUFFER_FLAG_END_OF_STREAM = MediaCodec.BUFFER_FLAG_END_OF_STREAM;
  /** Indicates that a buffer has supplemental data. */
  public static final int BUFFER_FLAG_HAS_SUPPLEMENTAL_DATA = 1 << 28; // 0x10000000
  /** Indicates that a buffer is known to contain the last media sample of the stream. */
  public static final int BUFFER_FLAG_LAST_SAMPLE = 1 << 29; // 0x20000000
  /** Indicates that a buffer is (at least partially) encrypted. */
  public static final int BUFFER_FLAG_ENCRYPTED = 1 << 30; // 0x40000000
  /** Indicates that a buffer should be decoded but not rendered. */
  public static final int BUFFER_FLAG_DECODE_ONLY = 1 << 31; // 0x80000000

  /**
   * Video decoder output modes. Possible modes are {@link #VIDEO_OUTPUT_MODE_NONE}, {@link
   * #VIDEO_OUTPUT_MODE_YUV} and {@link #VIDEO_OUTPUT_MODE_SURFACE_YUV}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef(value = {VIDEO_OUTPUT_MODE_NONE, VIDEO_OUTPUT_MODE_YUV, VIDEO_OUTPUT_MODE_SURFACE_YUV})
  public @interface VideoOutputMode {}
  /** Video decoder output mode is not set. */
  public static final int VIDEO_OUTPUT_MODE_NONE = -1;
  /** Video decoder output mode that outputs raw 4:2:0 YUV planes. */
  public static final int VIDEO_OUTPUT_MODE_YUV = 0;
  /** Video decoder output mode that renders 4:2:0 YUV planes directly to a surface. */
  public static final int VIDEO_OUTPUT_MODE_SURFACE_YUV = 1;

  /**
   * Video scaling modes for {@link MediaCodec}-based renderers. One of {@link
   * #VIDEO_SCALING_MODE_SCALE_TO_FIT} or {@link #VIDEO_SCALING_MODE_SCALE_TO_FIT_WITH_CROPPING}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef(value = {VIDEO_SCALING_MODE_SCALE_TO_FIT, VIDEO_SCALING_MODE_SCALE_TO_FIT_WITH_CROPPING})
  public @interface VideoScalingMode {}
  /** See {@link MediaCodec#VIDEO_SCALING_MODE_SCALE_TO_FIT}. */
  public static final int VIDEO_SCALING_MODE_SCALE_TO_FIT =
      MediaCodec.VIDEO_SCALING_MODE_SCALE_TO_FIT;
  /** See {@link MediaCodec#VIDEO_SCALING_MODE_SCALE_TO_FIT_WITH_CROPPING}. */
  public static final int VIDEO_SCALING_MODE_SCALE_TO_FIT_WITH_CROPPING =
      MediaCodec.VIDEO_SCALING_MODE_SCALE_TO_FIT_WITH_CROPPING;
  /** A default video scaling mode for {@link MediaCodec}-based renderers. */
  public static final int VIDEO_SCALING_MODE_DEFAULT = VIDEO_SCALING_MODE_SCALE_TO_FIT;

  /**
   * Track selection flags. Possible flag values are {@link #SELECTION_FLAG_DEFAULT}, {@link
   * #SELECTION_FLAG_FORCED} and {@link #SELECTION_FLAG_AUTOSELECT}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef(
      flag = true,
      value = {SELECTION_FLAG_DEFAULT, SELECTION_FLAG_FORCED, SELECTION_FLAG_AUTOSELECT})
  public @interface SelectionFlags {}
  /** Indicates that the track should be selected if user preferences do not state otherwise. */
  public static final int SELECTION_FLAG_DEFAULT = 1;
  /**
   * Indicates that the track should be selected if its language matches the language of the
   * selected audio track and user preferences do not state otherwise. Only applies to text tracks.
   *
   * <p>Tracks with this flag generally provide translation for elements that don't match the
   * declared language of the selected audio track (e.g. speech in an alien language). See <a
   * href="https://partnerhelp.netflixstudios.com/hc/en-us/articles/217558918">Netflix's summary</a>
   * for more info.
   */
  public static final int SELECTION_FLAG_FORCED = 1 << 1; // 2
  /**
   * Indicates that the player may choose to play the track in absence of an explicit user
   * preference.
   */
  public static final int SELECTION_FLAG_AUTOSELECT = 1 << 2; // 4

  /** Represents an undetermined language as an ISO 639-2 language code. */
  public static final String LANGUAGE_UNDETERMINED = "und";

  /**
   * Represents a streaming or other media type. One of {@link #TYPE_DASH}, {@link #TYPE_SS}, {@link
   * #TYPE_HLS}, {@link #TYPE_RTSP} or {@link #TYPE_OTHER}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({TYPE_DASH, TYPE_SS, TYPE_HLS, TYPE_RTSP, TYPE_OTHER})
  public @interface ContentType {}
  /** Value returned by {@link Util#inferContentType(String)} for DASH manifests. */
  public static final int TYPE_DASH = 0;
  /** Value returned by {@link Util#inferContentType(String)} for Smooth Streaming manifests. */
  public static final int TYPE_SS = 1;
  /** Value returned by {@link Util#inferContentType(String)} for HLS manifests. */
  public static final int TYPE_HLS = 2;
  /** Value returned by {@link Util#inferContentType(String)} for RTSP. */
  public static final int TYPE_RTSP = 3;
  /**
   * Value returned by {@link Util#inferContentType(String)} for files other than DASH, HLS or
   * Smooth Streaming manifests, or RTSP URIs.
   */
  public static final int TYPE_OTHER = 4;

  /** A return value for methods where the end of an input was encountered. */
  public static final int RESULT_END_OF_INPUT = -1;
  /**
   * A return value for methods where the length of parsed data exceeds the maximum length allowed.
   */
  public static final int RESULT_MAX_LENGTH_EXCEEDED = -2;
  /** A return value for methods where nothing was read. */
  public static final int RESULT_NOTHING_READ = -3;
  /** A return value for methods where a buffer was read. */
  public static final int RESULT_BUFFER_READ = -4;
  /** A return value for methods where a format was read. */
  public static final int RESULT_FORMAT_READ = -5;

  /**
   * Represents a type of data. May be one of {@link #DATA_TYPE_UNKNOWN}, {@link #DATA_TYPE_MEDIA},
   * {@link #DATA_TYPE_MEDIA_INITIALIZATION}, {@link #DATA_TYPE_DRM}, {@link #DATA_TYPE_MANIFEST},
   * {@link #DATA_TYPE_TIME_SYNCHRONIZATION}, {@link #DATA_TYPE_AD}, or {@link
   * #DATA_TYPE_MEDIA_PROGRESSIVE_LIVE}. May also be an app-defined value (see {@link
   * #DATA_TYPE_CUSTOM_BASE}).
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef(
      open = true,
      value = {
        DATA_TYPE_UNKNOWN,
        DATA_TYPE_MEDIA,
        DATA_TYPE_MEDIA_INITIALIZATION,
        DATA_TYPE_DRM,
        DATA_TYPE_MANIFEST,
        DATA_TYPE_TIME_SYNCHRONIZATION,
        DATA_TYPE_AD,
        DATA_TYPE_MEDIA_PROGRESSIVE_LIVE
      })
  public @interface DataType {}
  /** A data type constant for data of unknown or unspecified type. */
  public static final int DATA_TYPE_UNKNOWN = 0;
  /** A data type constant for media, typically containing media samples. */
  public static final int DATA_TYPE_MEDIA = 1;
  /** A data type constant for media, typically containing only initialization data. */
  public static final int DATA_TYPE_MEDIA_INITIALIZATION = 2;
  /** A data type constant for drm or encryption data. */
  public static final int DATA_TYPE_DRM = 3;
  /** A data type constant for a manifest file. */
  public static final int DATA_TYPE_MANIFEST = 4;
  /** A data type constant for time synchronization data. */
  public static final int DATA_TYPE_TIME_SYNCHRONIZATION = 5;
  /** A data type constant for ads loader data. */
  public static final int DATA_TYPE_AD = 6;
  /**
   * A data type constant for live progressive media streams, typically containing media samples.
   */
  public static final int DATA_TYPE_MEDIA_PROGRESSIVE_LIVE = 7;
  /**
   * Applications or extensions may define custom {@code DATA_TYPE_*} constants greater than or
   * equal to this value.
   */
  public static final int DATA_TYPE_CUSTOM_BASE = 10000;

  /** A type constant for tracks of unknown type. */
  public static final int TRACK_TYPE_UNKNOWN = -1;
  /** A type constant for tracks of some default type, where the type itself is unknown. */
  public static final int TRACK_TYPE_DEFAULT = 0;
  /** A type constant for audio tracks. */
  public static final int TRACK_TYPE_AUDIO = 1;
  /** A type constant for video tracks. */
  public static final int TRACK_TYPE_VIDEO = 2;
  /** A type constant for text tracks. */
  public static final int TRACK_TYPE_TEXT = 3;
  /** A type constant for image tracks. */
  public static final int TRACK_TYPE_IMAGE = 4;
  /** A type constant for metadata tracks. */
  public static final int TRACK_TYPE_METADATA = 5;
  /** A type constant for camera motion tracks. */
  public static final int TRACK_TYPE_CAMERA_MOTION = 6;
  /** A type constant for a fake or empty track. */
  public static final int TRACK_TYPE_NONE = 7;
  /**
   * Applications or extensions may define custom {@code TRACK_TYPE_*} constants greater than or
   * equal to this value.
   */
  public static final int TRACK_TYPE_CUSTOM_BASE = 10000;

  /** A selection reason constant for selections whose reasons are unknown or unspecified. */
  public static final int SELECTION_REASON_UNKNOWN = 0;
  /** A selection reason constant for an initial track selection. */
  public static final int SELECTION_REASON_INITIAL = 1;
  /** A selection reason constant for an manual (i.e. user initiated) track selection. */
  public static final int SELECTION_REASON_MANUAL = 2;
  /** A selection reason constant for an adaptive track selection. */
  public static final int SELECTION_REASON_ADAPTIVE = 3;
  /** A selection reason constant for a trick play track selection. */
  public static final int SELECTION_REASON_TRICK_PLAY = 4;
  /**
   * Applications or extensions may define custom {@code SELECTION_REASON_*} constants greater than
   * or equal to this value.
   */
  public static final int SELECTION_REASON_CUSTOM_BASE = 10000;

  /** A default size in bytes for an individual allocation that forms part of a larger buffer. */
  public static final int DEFAULT_BUFFER_SEGMENT_SIZE = 64 * 1024;

  /** A default seek back increment, in milliseconds. */
  public static final long DEFAULT_SEEK_BACK_INCREMENT_MS = 5000;
  /** A default seek forward increment, in milliseconds. */
  public static final long DEFAULT_SEEK_FORWARD_INCREMENT_MS = 15_000;

  /**
   * A default maximum position for which a seek to previous will seek to the previous window, in
   * milliseconds.
   */
  public static final int DEFAULT_MAX_SEEK_TO_PREVIOUS_POSITION_MS = 3000;

  /** "cenc" scheme type name as defined in ISO/IEC 23001-7:2016. */
  @SuppressWarnings("ConstantField")
  public static final String CENC_TYPE_cenc = "cenc";

  /** "cbc1" scheme type name as defined in ISO/IEC 23001-7:2016. */
  @SuppressWarnings("ConstantField")
  public static final String CENC_TYPE_cbc1 = "cbc1";

  /** "cens" scheme type name as defined in ISO/IEC 23001-7:2016. */
  @SuppressWarnings("ConstantField")
  public static final String CENC_TYPE_cens = "cens";

  /** "cbcs" scheme type name as defined in ISO/IEC 23001-7:2016. */
  @SuppressWarnings("ConstantField")
  public static final String CENC_TYPE_cbcs = "cbcs";

  /**
   * The Nil UUID as defined by <a
   * href="https://tools.ietf.org/html/rfc4122#section-4.1.7">RFC4122</a>.
   */
  public static final UUID UUID_NIL = new UUID(0L, 0L);

  /**
   * UUID for the W3C <a
   * href="https://w3c.github.io/encrypted-media/format-registry/initdata/cenc.html">Common PSSH
   * box</a>.
   */
  public static final UUID COMMON_PSSH_UUID = new UUID(0x1077EFECC0B24D02L, 0xACE33C1E52E2FB4BL);

  /**
   * UUID for the ClearKey DRM scheme.
   *
   * <p>ClearKey is supported on Android devices running Android 5.0 (API Level 21) and up.
   */
  public static final UUID CLEARKEY_UUID = new UUID(0xE2719D58A985B3C9L, 0x781AB030AF78D30EL);

  /**
   * UUID for the Widevine DRM scheme.
   *
   * <p>Widevine is supported on Android devices running Android 4.3 (API Level 18) and up.
   */
  public static final UUID WIDEVINE_UUID = new UUID(0xEDEF8BA979D64ACEL, 0xA3C827DCD51D21EDL);

  /**
   * UUID for the PlayReady DRM scheme.
   *
   * <p>PlayReady is supported on all AndroidTV devices. Note that most other Android devices do not
   * provide PlayReady support.
   */
  public static final UUID PLAYREADY_UUID = new UUID(0x9A04F07998404286L, 0xAB92E65BE0885F95L);

  //weiguo
  //ChinaDRM software
  //public static final UUID CHINADRM_UUID = new UUID(0x70c1db9f66ae4127L, 0xbfc0bb1981694b67L);
  //ChinaDRM TEE
  public static final UUID CHINADRM_UUID = new UUID(0x70c1db9f66ae4127L, 0xbfc0bb1981694b66L);
  public static final UUID WISEPLAY_UUID = new UUID(0X3D5E6D359B9A41E8L, 0XB843DD3C6E72C42CL);
  /** "sm4c" scheme type name as defined in ISO/IEC 23001-7:2016. */
  public static final String CENC_TYPE_sm4c = "sm4c";
  /** "sm4s" scheme type name as defined in ISO/IEC 23001-7:2016. */
  public static final String CENC_TYPE_sm4s = "sm4s";


  /** @deprecated Use {@code Renderer.MSG_SET_VIDEO_OUTPUT}. */
  @Deprecated public static final int MSG_SET_SURFACE = 1;

  /** @deprecated Use {@code Renderer.MSG_SET_VOLUME}. */
  @Deprecated public static final int MSG_SET_VOLUME = 2;

  /** @deprecated Use {@code Renderer.MSG_SET_AUDIO_ATTRIBUTES}. */
  @Deprecated public static final int MSG_SET_AUDIO_ATTRIBUTES = 3;

  /** @deprecated Use {@code Renderer.MSG_SET_SCALING_MODE}. */
  @Deprecated public static final int MSG_SET_SCALING_MODE = 4;

  /** @deprecated Use {@code Renderer.MSG_SET_AUX_EFFECT_INFO}. */
  @Deprecated public static final int MSG_SET_AUX_EFFECT_INFO = 5;

  /** @deprecated Use {@code Renderer.MSG_SET_VIDEO_FRAME_METADATA_LISTENER}. */
  @Deprecated public static final int MSG_SET_VIDEO_FRAME_METADATA_LISTENER = 6;

  /** @deprecated Use {@code Renderer.MSG_SET_CAMERA_MOTION_LISTENER}. */
  @Deprecated public static final int MSG_SET_CAMERA_MOTION_LISTENER = 7;

  /** @deprecated Use {@code Renderer.MSG_CUSTOM_BASE}. */
  @Deprecated public static final int MSG_CUSTOM_BASE = 10000;

  /**
   * The stereo mode for 360/3D/VR videos. One of {@link Format#NO_VALUE}, {@link
   * #STEREO_MODE_MONO}, {@link #STEREO_MODE_TOP_BOTTOM}, {@link #STEREO_MODE_LEFT_RIGHT} or {@link
   * #STEREO_MODE_STEREO_MESH}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    Format.NO_VALUE,
    STEREO_MODE_MONO,
    STEREO_MODE_TOP_BOTTOM,
    STEREO_MODE_LEFT_RIGHT,
    STEREO_MODE_STEREO_MESH
  })
  public @interface StereoMode {}
  /** Indicates Monoscopic stereo layout, used with 360/3D/VR videos. */
  public static final int STEREO_MODE_MONO = 0;
  /** Indicates Top-Bottom stereo layout, used with 360/3D/VR videos. */
  public static final int STEREO_MODE_TOP_BOTTOM = 1;
  /** Indicates Left-Right stereo layout, used with 360/3D/VR videos. */
  public static final int STEREO_MODE_LEFT_RIGHT = 2;
  /**
   * Indicates a stereo layout where the left and right eyes have separate meshes, used with
   * 360/3D/VR videos.
   */
  public static final int STEREO_MODE_STEREO_MESH = 3;

  /**
   * Video colorspaces. One of {@link Format#NO_VALUE}, {@link #COLOR_SPACE_BT709}, {@link
   * #COLOR_SPACE_BT601} or {@link #COLOR_SPACE_BT2020}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({Format.NO_VALUE, COLOR_SPACE_BT709, COLOR_SPACE_BT601, COLOR_SPACE_BT2020})
  public @interface ColorSpace {}
  /** @see MediaFormat#COLOR_STANDARD_BT709 */
  public static final int COLOR_SPACE_BT709 = MediaFormat.COLOR_STANDARD_BT709;
  /** @see MediaFormat#COLOR_STANDARD_BT601_PAL */
  public static final int COLOR_SPACE_BT601 = MediaFormat.COLOR_STANDARD_BT601_PAL;
  /** @see MediaFormat#COLOR_STANDARD_BT2020 */
  public static final int COLOR_SPACE_BT2020 = MediaFormat.COLOR_STANDARD_BT2020;

  /**
   * Video color transfer characteristics. One of {@link Format#NO_VALUE}, {@link
   * #COLOR_TRANSFER_SDR}, {@link #COLOR_TRANSFER_ST2084} or {@link #COLOR_TRANSFER_HLG}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({Format.NO_VALUE, COLOR_TRANSFER_SDR, COLOR_TRANSFER_ST2084, COLOR_TRANSFER_HLG})
  public @interface ColorTransfer {}
  /** @see MediaFormat#COLOR_TRANSFER_SDR_VIDEO */
  public static final int COLOR_TRANSFER_SDR = MediaFormat.COLOR_TRANSFER_SDR_VIDEO;
  /** @see MediaFormat#COLOR_TRANSFER_ST2084 */
  public static final int COLOR_TRANSFER_ST2084 = MediaFormat.COLOR_TRANSFER_ST2084;
  /** @see MediaFormat#COLOR_TRANSFER_HLG */
  public static final int COLOR_TRANSFER_HLG = MediaFormat.COLOR_TRANSFER_HLG;

  /**
   * Video color range. One of {@link Format#NO_VALUE}, {@link #COLOR_RANGE_LIMITED} or {@link
   * #COLOR_RANGE_FULL}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({Format.NO_VALUE, COLOR_RANGE_LIMITED, COLOR_RANGE_FULL})
  public @interface ColorRange {}
  /** @see MediaFormat#COLOR_RANGE_LIMITED */
  public static final int COLOR_RANGE_LIMITED = MediaFormat.COLOR_RANGE_LIMITED;
  /** @see MediaFormat#COLOR_RANGE_FULL */
  public static final int COLOR_RANGE_FULL = MediaFormat.COLOR_RANGE_FULL;

  /** Video projection types. */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    Format.NO_VALUE,
    PROJECTION_RECTANGULAR,
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_CUBEMAP,
    PROJECTION_MESH
  })
  public @interface Projection {}
  /** Conventional rectangular projection. */
  public static final int PROJECTION_RECTANGULAR = 0;
  /** Equirectangular spherical projection. */
  public static final int PROJECTION_EQUIRECTANGULAR = 1;
  /** Cube map projection. */
  public static final int PROJECTION_CUBEMAP = 2;
  /** 3-D mesh projection. */
  public static final int PROJECTION_MESH = 3;

  /**
   * Priority for media playback.
   *
   * <p>Larger values indicate higher priorities.
   */
  public static final int PRIORITY_PLAYBACK = 0;

  /**
   * Priority for media downloading.
   *
   * <p>Larger values indicate higher priorities.
   */
  public static final int PRIORITY_DOWNLOAD = PRIORITY_PLAYBACK - 1000;

  /**
   * Network connection type. One of {@link #NETWORK_TYPE_UNKNOWN}, {@link #NETWORK_TYPE_OFFLINE},
   * {@link #NETWORK_TYPE_WIFI}, {@link #NETWORK_TYPE_2G}, {@link #NETWORK_TYPE_3G}, {@link
   * #NETWORK_TYPE_4G}, {@link #NETWORK_TYPE_5G_SA}, {@link #NETWORK_TYPE_5G_NSA}, {@link
   * #NETWORK_TYPE_CELLULAR_UNKNOWN}, {@link #NETWORK_TYPE_ETHERNET} or {@link #NETWORK_TYPE_OTHER}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    NETWORK_TYPE_UNKNOWN,
    NETWORK_TYPE_OFFLINE,
    NETWORK_TYPE_WIFI,
    NETWORK_TYPE_2G,
    NETWORK_TYPE_3G,
    NETWORK_TYPE_4G,
    NETWORK_TYPE_5G_SA,
    NETWORK_TYPE_5G_NSA,
    NETWORK_TYPE_CELLULAR_UNKNOWN,
    NETWORK_TYPE_ETHERNET,
    NETWORK_TYPE_OTHER
  })
  public @interface NetworkType {}
  /** Unknown network type. */
  public static final int NETWORK_TYPE_UNKNOWN = 0;
  /** No network connection. */
  public static final int NETWORK_TYPE_OFFLINE = 1;
  /** Network type for a Wifi connection. */
  public static final int NETWORK_TYPE_WIFI = 2;
  /** Network type for a 2G cellular connection. */
  public static final int NETWORK_TYPE_2G = 3;
  /** Network type for a 3G cellular connection. */
  public static final int NETWORK_TYPE_3G = 4;
  /** Network type for a 4G cellular connection. */
  public static final int NETWORK_TYPE_4G = 5;
  /** Network type for a 5G stand-alone (SA) cellular connection. */
  public static final int NETWORK_TYPE_5G_SA = 9;
  /** Network type for a 5G non-stand-alone (NSA) cellular connection. */
  public static final int NETWORK_TYPE_5G_NSA = 10;
  /**
   * Network type for cellular connections which cannot be mapped to one of {@link
   * #NETWORK_TYPE_2G}, {@link #NETWORK_TYPE_3G}, or {@link #NETWORK_TYPE_4G}.
   */
  public static final int NETWORK_TYPE_CELLULAR_UNKNOWN = 6;
  /** Network type for an Ethernet connection. */
  public static final int NETWORK_TYPE_ETHERNET = 7;
  /** Network type for other connections which are not Wifi or cellular (e.g. VPN, Bluetooth). */
  public static final int NETWORK_TYPE_OTHER = 8;

  /**
   * Mode specifying whether the player should hold a WakeLock and a WifiLock. One of {@link
   * #WAKE_MODE_NONE}, {@link #WAKE_MODE_LOCAL} or {@link #WAKE_MODE_NETWORK}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({WAKE_MODE_NONE, WAKE_MODE_LOCAL, WAKE_MODE_NETWORK})
  public @interface WakeMode {}
  /**
   * A wake mode that will not cause the player to hold any locks.
   *
   * <p>This is suitable for applications that do not play media with the screen off.
   */
  public static final int WAKE_MODE_NONE = 0;
  /**
   * A wake mode that will cause the player to hold a {@link android.os.PowerManager.WakeLock}
   * during playback.
   *
   * <p>This is suitable for applications that play media with the screen off and do not load media
   * over wifi.
   */
  public static final int WAKE_MODE_LOCAL = 1;
  /**
   * A wake mode that will cause the player to hold a {@link android.os.PowerManager.WakeLock} and a
   * {@link android.net.wifi.WifiManager.WifiLock} during playback.
   *
   * <p>This is suitable for applications that play media with the screen off and may load media
   * over wifi.
   */
  public static final int WAKE_MODE_NETWORK = 2;

  /**
   * Track role flags. Possible flag values are {@link #ROLE_FLAG_MAIN}, {@link
   * #ROLE_FLAG_ALTERNATE}, {@link #ROLE_FLAG_SUPPLEMENTARY}, {@link #ROLE_FLAG_COMMENTARY}, {@link
   * #ROLE_FLAG_DUB}, {@link #ROLE_FLAG_EMERGENCY}, {@link #ROLE_FLAG_CAPTION}, {@link
   * #ROLE_FLAG_SUBTITLE}, {@link #ROLE_FLAG_SIGN}, {@link #ROLE_FLAG_DESCRIBES_VIDEO}, {@link
   * #ROLE_FLAG_DESCRIBES_MUSIC_AND_SOUND}, {@link #ROLE_FLAG_ENHANCED_DIALOG_INTELLIGIBILITY},
   * {@link #ROLE_FLAG_TRANSCRIBES_DIALOG}, {@link #ROLE_FLAG_EASY_TO_READ} and {@link
   * #ROLE_FLAG_TRICK_PLAY}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef(
      flag = true,
      value = {
        ROLE_FLAG_MAIN,
        ROLE_FLAG_ALTERNATE,
        ROLE_FLAG_SUPPLEMENTARY,
        ROLE_FLAG_COMMENTARY,
        ROLE_FLAG_DUB,
        ROLE_FLAG_EMERGENCY,
        ROLE_FLAG_CAPTION,
        ROLE_FLAG_SUBTITLE,
        ROLE_FLAG_SIGN,
        ROLE_FLAG_DESCRIBES_VIDEO,
        ROLE_FLAG_DESCRIBES_MUSIC_AND_SOUND,
        ROLE_FLAG_ENHANCED_DIALOG_INTELLIGIBILITY,
        ROLE_FLAG_TRANSCRIBES_DIALOG,
        ROLE_FLAG_EASY_TO_READ,
        ROLE_FLAG_TRICK_PLAY
      })
  public @interface RoleFlags {}
  /** Indicates a main track. */
  public static final int ROLE_FLAG_MAIN = 1;
  /**
   * Indicates an alternate track. For example a video track recorded from an different view point
   * than the main track(s).
   */
  public static final int ROLE_FLAG_ALTERNATE = 1 << 1;
  /**
   * Indicates a supplementary track, meaning the track has lower importance than the main track(s).
   * For example a video track that provides a visual accompaniment to a main audio track.
   */
  public static final int ROLE_FLAG_SUPPLEMENTARY = 1 << 2;
  /** Indicates the track contains commentary, for example from the director. */
  public static final int ROLE_FLAG_COMMENTARY = 1 << 3;
  /**
   * Indicates the track is in a different language from the original, for example dubbed audio or
   * translated captions.
   */
  public static final int ROLE_FLAG_DUB = 1 << 4;
  /** Indicates the track contains information about a current emergency. */
  public static final int ROLE_FLAG_EMERGENCY = 1 << 5;
  /**
   * Indicates the track contains captions. This flag may be set on video tracks to indicate the
   * presence of burned in captions.
   */
  public static final int ROLE_FLAG_CAPTION = 1 << 6;
  /**
   * Indicates the track contains subtitles. This flag may be set on video tracks to indicate the
   * presence of burned in subtitles.
   */
  public static final int ROLE_FLAG_SUBTITLE = 1 << 7;
  /** Indicates the track contains a visual sign-language interpretation of an audio track. */
  public static final int ROLE_FLAG_SIGN = 1 << 8;
  /** Indicates the track contains an audio or textual description of a video track. */
  public static final int ROLE_FLAG_DESCRIBES_VIDEO = 1 << 9;
  /** Indicates the track contains a textual description of music and sound. */
  public static final int ROLE_FLAG_DESCRIBES_MUSIC_AND_SOUND = 1 << 10;
  /** Indicates the track is designed for improved intelligibility of dialogue. */
  public static final int ROLE_FLAG_ENHANCED_DIALOG_INTELLIGIBILITY = 1 << 11;
  /** Indicates the track contains a transcription of spoken dialog. */
  public static final int ROLE_FLAG_TRANSCRIBES_DIALOG = 1 << 12;
  /** Indicates the track contains a text that has been edited for ease of reading. */
  public static final int ROLE_FLAG_EASY_TO_READ = 1 << 13;
  /** Indicates the track is intended for trick play. */
  public static final int ROLE_FLAG_TRICK_PLAY = 1 << 14;

  /**
   * Level of renderer support for a format. One of {@link #FORMAT_HANDLED}, {@link
   * #FORMAT_EXCEEDS_CAPABILITIES}, {@link #FORMAT_UNSUPPORTED_DRM}, {@link
   * #FORMAT_UNSUPPORTED_SUBTYPE} or {@link #FORMAT_UNSUPPORTED_TYPE}.
   */
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    FORMAT_HANDLED,
    FORMAT_EXCEEDS_CAPABILITIES,
    FORMAT_UNSUPPORTED_DRM,
    FORMAT_UNSUPPORTED_SUBTYPE,
    FORMAT_UNSUPPORTED_TYPE
  })
  public static @interface FormatSupport {}
  // TODO(b/172315872) Renderer was a link. Link to equivalent concept or remove @code.
  /** The {@code Renderer} is capable of rendering the format. */
  public static final int FORMAT_HANDLED = 0b100;
  /**
   * The {@code Renderer} is capable of rendering formats with the same MIME type, but the
   * properties of the format exceed the renderer's capabilities. There is a chance the renderer
   * will be able to play the format in practice because some renderers report their capabilities
   * conservatively, but the expected outcome is that playback will fail.
   *
   * <p>Example: The {@code Renderer} is capable of rendering H264 and the format's MIME type is
   * {@code MimeTypes#VIDEO_H264}, but the format's resolution exceeds the maximum limit supported
   * by the underlying H264 decoder.
   */
  public static final int FORMAT_EXCEEDS_CAPABILITIES = 0b011;
  /**
   * The {@code Renderer} is capable of rendering formats with the same MIME type, but is not
   * capable of rendering the format because the format's drm protection is not supported.
   *
   * <p>Example: The {@code Renderer} is capable of rendering H264 and the format's MIME type is
   * {@link MimeTypes#VIDEO_H264}, but the format indicates PlayReady drm protection whereas the
   * renderer only supports Widevine.
   */
  public static final int FORMAT_UNSUPPORTED_DRM = 0b010;
  /**
   * The {@code Renderer} is a general purpose renderer for formats of the same top-level type, but
   * is not capable of rendering the format or any other format with the same MIME type because the
   * sub-type is not supported.
   *
   * <p>Example: The {@code Renderer} is a general purpose audio renderer and the format's MIME type
   * matches audio/[subtype], but there does not exist a suitable decoder for [subtype].
   */
  public static final int FORMAT_UNSUPPORTED_SUBTYPE = 0b001;
  /**
   * The {@code Renderer} is not capable of rendering the format, either because it does not support
   * the format's top-level type, or because it's a specialized renderer for a different MIME type.
   *
   * <p>Example: The {@code Renderer} is a general purpose video renderer, but the format has an
   * audio MIME type.
   */
  public static final int FORMAT_UNSUPPORTED_TYPE = 0b000;
  /**
   * Converts a time in microseconds to the corresponding time in milliseconds, preserving {@link
   * #TIME_UNSET} and {@link #TIME_END_OF_SOURCE} values.
   *
   * @param timeUs The time in microseconds.
   * @return The corresponding time in milliseconds.
   */
  public static long usToMs(long timeUs) {
    return (timeUs == TIME_UNSET || timeUs == TIME_END_OF_SOURCE) ? timeUs : (timeUs / 1000);
  }

  /**
   * Converts a time in milliseconds to the corresponding time in microseconds, preserving {@link
   * #TIME_UNSET} values and {@link #TIME_END_OF_SOURCE} values.
   *
   * @param timeMs The time in milliseconds.
   * @return The corresponding time in microseconds.
   */
  public static long msToUs(long timeMs) {
    return (timeMs == TIME_UNSET || timeMs == TIME_END_OF_SOURCE) ? timeMs : (timeMs * 1000);
  }

  /**
   * Returns a newly generated audio session identifier, or {@link AudioManager#ERROR} if an error
   * occurred in which case audio playback may fail.
   *
   * @see AudioManager#generateAudioSessionId()
   */
  @RequiresApi(21)
  public static int generateAudioSessionIdV21(Context context) {
    @Nullable
    AudioManager audioManager = ((AudioManager) context.getSystemService(Context.AUDIO_SERVICE));
    return audioManager == null ? AudioManager.ERROR : audioManager.generateAudioSessionId();
  }

  /**
   * Returns string representation of a {@link FormatSupport} flag.
   *
   * @param formatSupport A {@link FormatSupport} flag.
   * @return A string representation of the flag.
   */
  public static String getFormatSupportString(@FormatSupport int formatSupport) {
    switch (formatSupport) {
      case FORMAT_HANDLED:
        return "YES";
      case FORMAT_EXCEEDS_CAPABILITIES:
        return "NO_EXCEEDS_CAPABILITIES";
      case FORMAT_UNSUPPORTED_DRM:
        return "NO_UNSUPPORTED_DRM";
      case FORMAT_UNSUPPORTED_SUBTYPE:
        return "NO_UNSUPPORTED_TYPE";
      case FORMAT_UNSUPPORTED_TYPE:
        return "NO";
      default:
        throw new IllegalStateException();
    }
  }

  // Copy of relevant error codes defined in MediaDrm.ErrorCodes from API 31.
  // TODO (internal b/192337376): Remove once ExoPlayer depends on SDK 31.
  private static final int ERROR_KEY_EXPIRED = 2;
  private static final int ERROR_INSUFFICIENT_OUTPUT_PROTECTION = 4;
  private static final int ERROR_INSUFFICIENT_SECURITY = 7;
  private static final int ERROR_FRAME_TOO_LARGE = 8;
  private static final int ERROR_CERTIFICATE_MALFORMED = 10;
  private static final int ERROR_INIT_DATA = 15;
  private static final int ERROR_KEY_NOT_LOADED = 16;
  private static final int ERROR_LICENSE_PARSE = 17;
  private static final int ERROR_LICENSE_POLICY = 18;
  private static final int ERROR_LICENSE_RELEASE = 19;
  private static final int ERROR_LICENSE_REQUEST_REJECTED = 20;
  private static final int ERROR_LICENSE_RESTORE = 21;
  private static final int ERROR_LICENSE_STATE = 22;
  private static final int ERROR_PROVISIONING_CERTIFICATE = 24;
  private static final int ERROR_PROVISIONING_CONFIG = 25;
  private static final int ERROR_PROVISIONING_PARSE = 26;
  private static final int ERROR_PROVISIONING_REQUEST_REJECTED = 27;
  private static final int ERROR_PROVISIONING_RETRY = 28;

  @PlaybackException.ErrorCode
  public static int getErrorCodeForMediaDrmErrorCode(int mediaDrmErrorCode) {
    switch (mediaDrmErrorCode) {
      case ERROR_PROVISIONING_CONFIG:
      case ERROR_PROVISIONING_PARSE:
      case ERROR_PROVISIONING_REQUEST_REJECTED:
      case ERROR_PROVISIONING_CERTIFICATE:
      case ERROR_PROVISIONING_RETRY:
        return PlaybackException.ERROR_CODE_DRM_PROVISIONING_FAILED;
      case ERROR_LICENSE_PARSE:
      case ERROR_LICENSE_RELEASE:
      case ERROR_LICENSE_REQUEST_REJECTED:
      case ERROR_LICENSE_RESTORE:
      case ERROR_LICENSE_STATE:
      case ERROR_CERTIFICATE_MALFORMED:
        return PlaybackException.ERROR_CODE_DRM_LICENSE_ACQUISITION_FAILED;
      case ERROR_LICENSE_POLICY:
      case ERROR_INSUFFICIENT_OUTPUT_PROTECTION:
      case ERROR_INSUFFICIENT_SECURITY:
      case ERROR_KEY_EXPIRED:
      case ERROR_KEY_NOT_LOADED:
        return PlaybackException.ERROR_CODE_DRM_DISALLOWED_OPERATION;
      case ERROR_INIT_DATA:
      case ERROR_FRAME_TOO_LARGE:
        return PlaybackException.ERROR_CODE_DRM_CONTENT_ERROR;
      default:
        return PlaybackException.ERROR_CODE_DRM_SYSTEM_ERROR;
    }
  }
}
