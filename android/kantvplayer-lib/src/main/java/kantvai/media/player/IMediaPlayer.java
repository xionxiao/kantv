/*
 * Copyright (C) 2013-2014 Bilibili
 * Copyright (C) 2013-2014 Zhang Rui <bbcallen@gmail.com>
 * Copyright (c) Project KanTV. 2021-2023. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

package kantvai.media.player;

import android.annotation.TargetApi;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.view.Surface;
import android.view.SurfaceHolder;

import java.io.FileDescriptor;
import java.io.IOException;
import java.util.Map;

import kantvai.media.player.bean.TrackInfoBean;
import kantvai.media.player.misc.IMediaDataSource;
import kantvai.media.player.misc.ITrackInfo;

public interface IMediaPlayer {
    /*
     * Do not change these values without updating their counterparts in native
     */
    int MEDIA_INFO_UNKNOWN = 1;
    int MEDIA_INFO_STARTED_AS_NEXT = 2;
    int MEDIA_INFO_VIDEO_RENDERING_START = 3;
    int MEDIA_INFO_VIDEO_TRACK_LAGGING = 700;
    int MEDIA_INFO_BUFFERING_START = 701;
    int MEDIA_INFO_BUFFERING_END = 702;
    int MEDIA_INFO_NETWORK_BANDWIDTH = 703;
    int MEDIA_INFO_BAD_INTERLEAVING = 800;
    int MEDIA_INFO_NOT_SEEKABLE = 801;
    int MEDIA_INFO_METADATA_UPDATE = 802;
    int MEDIA_INFO_TIMED_TEXT_ERROR = 900;
    int MEDIA_INFO_UNSUPPORTED_SUBTITLE = 901;
    int MEDIA_INFO_SUBTITLE_TIMED_OUT = 902;

    int MEDIA_INFO_RECORDING_START = 2000;
    int MEDIA_INFO_RECORDING_STOP  = 2001;

    int MEDIA_INFO_ASR_START = 2024;
    int MEDIA_INFO_ASR_STOP  = 2025;
    int MEDIA_INFO_ASR_START_NORMAL = 2026;
    int MEDIA_INFO_ASR_STOP_NORMAL  = 2027;

    int MEDIA_INFO_VIDEO_ROTATION_CHANGED = 10001;
    int MEDIA_INFO_AUDIO_RENDERING_START  = 10002;
    int MEDIA_INFO_AUDIO_DECODED_START    = 10003;
    int MEDIA_INFO_VIDEO_DECODED_START    = 10004;
    int MEDIA_INFO_OPEN_INPUT             = 10005;
    int MEDIA_INFO_FIND_STREAM_INFO       = 10006;
    int MEDIA_INFO_COMPONENT_OPEN         = 10007;
    int MEDIA_INFO_VIDEO_SEEK_RENDERING_START = 10008;
    int MEDIA_INFO_AUDIO_SEEK_RENDERING_START = 10009;
    int MEDIA_INFO_MEDIA_ACCURATE_SEEK_COMPLETE = 10100;

    int MEDIA_ERROR_UNKNOWN = 1;
    int MEDIA_ERROR_SERVER_DIED = 100;
    int MEDIA_ERROR_NOT_VALID_FOR_PROGRESSIVE_PLAYBACK = 200;
    int MEDIA_ERROR_IO = -1004;
    int MEDIA_ERROR_MALFORMED = -1007;
    int MEDIA_ERROR_UNSUPPORTED = -1010;
    int MEDIA_ERROR_TIMED_OUT = -110;

    void setDisplay(SurfaceHolder sh);

    void setDataSource(Context context, Uri uri)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException;

    @TargetApi(Build.VERSION_CODES.ICE_CREAM_SANDWICH)
    void setDataSource(Context context, Uri uri, Map<String, String> headers)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException;

    void setDataSource(FileDescriptor fd)
            throws IOException, IllegalArgumentException, IllegalStateException;

    void setDataSource(String path)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException;

    void setVideoTitle(String title);

    String getVideoTitle();

    String getDataSource();

    void prepareAsync() throws IllegalStateException;

    void start() throws IllegalStateException;

    void stop() throws IllegalStateException;

    void pause() throws IllegalStateException;

    void setScreenOnWhilePlaying(boolean screenOn);

    int getVideoWidth();

    int getVideoHeight();

    boolean isPlaying();

    void seekTo(long msec) throws IllegalStateException;

    long getCurrentPosition();

    long getDuration();

    void release();

    void reset();

    void setVolume(float leftVolume, float rightVolume);

    int getAudioSessionId();

    IjkMediaInfo getMediaInfo();

    boolean isClearContent();
    String  getDrmInfo();
    void setDecryptMode(int decryptMode);
    long getBeginTime();
    long getPauseTime();
    long getPlayTime();
    void setDisableAudioTrack(boolean bDisable);
    void setDisableVideoTrack(boolean bDisable);
    void setEnableDumpVideoES(boolean bEnable);
    void setEnableDumpAudioES(boolean bEnable);

    void setEnableRecord(boolean bEnableRecord);
    void updateRecordingStatus(boolean bEnableRecord);
    boolean getEnableRecord();

    void setEnableASR(boolean bEnableASR);
    boolean getEnableASR();

    void setDurationDumpES(int durationDumpES);//per seconds
    String getVideoInfo();
    String getAudioInfo();
    String getVideoDecoderName();
    String getAudioDecoderName();

    void selectTrack(TrackInfoBean trackInfo, boolean isAudio);
    long getBitRate();

    @SuppressWarnings("EmptyMethod")
    @Deprecated
    void setLogEnabled(boolean enable);

    @Deprecated
    boolean isPlayable();

    void setOnPreparedListener(OnPreparedListener listener);

    void setOnCompletionListener(OnCompletionListener listener);

    void setOnBufferingUpdateListener(
            OnBufferingUpdateListener listener);

    void setOnSeekCompleteListener(
            OnSeekCompleteListener listener);

    void setOnVideoSizeChangedListener(
            OnVideoSizeChangedListener listener);

    void setOnErrorListener(OnErrorListener listener);

    void setOnInfoListener(OnInfoListener listener);

    void setOnTimedTextListener(OnTimedTextListener listener);

    void setOnTrackChangeListener(OnTrackChangeListener listener);

    /*--------------------
     * Listeners
     */
    interface OnPreparedListener {
        void onPrepared(IMediaPlayer mp);
    }

    interface OnCompletionListener {
        void onCompletion(IMediaPlayer mp);
    }

    interface OnBufferingUpdateListener {
        void onBufferingUpdate(IMediaPlayer mp, int percent);
    }

    interface OnSeekCompleteListener {
        void onSeekComplete(IMediaPlayer mp);
    }

    interface OnVideoSizeChangedListener {
        void onVideoSizeChanged(IMediaPlayer mp, int width, int height,
                                int sar_num, int sar_den);
    }

    interface OnErrorListener {
        boolean onError(IMediaPlayer mp, int what, int extra);
    }

    interface OnInfoListener {
        boolean onInfo(IMediaPlayer mp, int what, int extra);
    }

    interface OnTimedTextListener {
        void onTimedText(IMediaPlayer mp, IjkTimedText text);
    }

    interface OnTrackChangeListener {
        void onTrackChange(IMediaPlayer mp, Object  trackSelector, Object trackGroups, Object trackSelections);
    }

    interface OnOutsideListener {
        void onAction (int action, long extra);
    }

    /*--------------------
     * Optional
     */
    void setAudioStreamType(int streamtype);

    @Deprecated
    void setKeepInBackground(boolean keepInBackground);

    int getVideoSarNum();

    int getVideoSarDen();

    @Deprecated
    void setWakeMode(Context context, int mode);

    void setLooping(boolean looping);

    boolean isLooping();

    /*--------------------
     * AndroidMediaPlayer: JELLY_BEAN
     */
    ITrackInfo[] getTrackInfo();

    /*--------------------
     * AndroidMediaPlayer: ICE_CREAM_SANDWICH:
     */
    void setSurface(Surface surface);

    /*--------------------
     * AndroidMediaPlayer: M:
     */
    void setDataSource(IMediaDataSource mediaDataSource);
}
