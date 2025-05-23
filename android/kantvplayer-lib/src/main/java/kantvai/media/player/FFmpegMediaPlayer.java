/*
 * Copyright (C) 2013-2014 Bilibili
 * Copyright (C) 2013-2014 Zhang Rui <bbcallen@gmail.com>
 * Copyright (c) Project KanTV. 2021-2023
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

import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.content.ContentResolver;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.SurfaceTexture;
import android.graphics.Rect;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.ParcelFileDescriptor;
import android.os.PowerManager;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;

import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.lang.reflect.Field;
import java.security.InvalidParameterException;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Map;
import java.util.TimeZone;

import kantvai.media.player.annotations.AccessedByNative;
import kantvai.media.player.annotations.CalledByNative;
import kantvai.media.player.bean.TrackInfoBean;
import kantvai.media.player.misc.IAndroidIO;
import kantvai.media.player.misc.IMediaDataSource;
import kantvai.media.player.misc.IMediaFormat;
import kantvai.media.player.misc.ITrackInfo;
import kantvai.media.player.misc.IjkTrackInfo;

/**
 * @author bbcallen
 * <p>
 * Java wrapper of ffplay.
 */
public final class FFmpegMediaPlayer extends AbstractMediaPlayer {
    private final static String TAG = FFmpegMediaPlayer.class.getName();

    private static final int MEDIA_NOP = 0; // interface test message
    private static final int MEDIA_PREPARED = 1;
    private static final int MEDIA_PLAYBACK_COMPLETE = 2;
    private static final int MEDIA_BUFFERING_UPDATE = 3;
    private static final int MEDIA_SEEK_COMPLETE = 4;
    private static final int MEDIA_SET_VIDEO_SIZE = 5;
    private static final int MEDIA_TIMED_TEXT = 99;
    private static final int MEDIA_ERROR = 100;
    private static final int MEDIA_INFO = 200;

    protected static final int MEDIA_SET_VIDEO_SAR = 10001;

    //----------------------------------------
    // options
    public static final int IJK_LOG_UNKNOWN = 0;
    public static final int IJK_LOG_DEFAULT = 1;

    public static final int IJK_LOG_VERBOSE = 2;
    public static final int IJK_LOG_DEBUG = 3;
    public static final int IJK_LOG_INFO = 4;
    public static final int IJK_LOG_WARN = 5;
    public static final int IJK_LOG_ERROR = 6;
    public static final int IJK_LOG_FATAL = 7;
    public static final int IJK_LOG_SILENT = 8;

    public static final int OPT_CATEGORY_FORMAT = 1;
    public static final int OPT_CATEGORY_CODEC = 2;
    public static final int OPT_CATEGORY_SWS = 3;
    public static final int OPT_CATEGORY_PLAYER = 4;

    public static final int SDL_FCC_YV12 = 0x32315659; // YV12
    public static final int SDL_FCC_RV16 = 0x36315652; // RGB565
    public static final int SDL_FCC_RV32 = 0x32335652; // RGBX8888
    //----------------------------------------

    //----------------------------------------
    // properties
    public static final int PROP_FLOAT_VIDEO_DECODE_FRAMES_PER_SECOND = 10001;
    public static final int PROP_FLOAT_VIDEO_OUTPUT_FRAMES_PER_SECOND = 10002;
    public static final int FFP_PROP_FLOAT_PLAYBACK_RATE = 10003;
    public static final int FFP_PROP_FLOAT_DROP_FRAME_RATE = 10007;

    public static final int FFP_PROP_INT64_SELECTED_VIDEO_STREAM = 20001;
    public static final int FFP_PROP_INT64_SELECTED_AUDIO_STREAM = 20002;
    public static final int FFP_PROP_INT64_SELECTED_TIMEDTEXT_STREAM = 20011;

    public static final int FFP_PROP_INT64_VIDEO_DECODER = 20003;
    public static final int FFP_PROP_INT64_AUDIO_DECODER = 20004;
    public static final int FFP_PROPV_DECODER_UNKNOWN = 0;
    public static final int FFP_PROPV_DECODER_AVCODEC = 1;
    public static final int FFP_PROPV_DECODER_MEDIACODEC = 2;
    public static final int FFP_PROPV_DECODER_VIDEOTOOLBOX = 3;
    public static final int FFP_PROP_INT64_VIDEO_CACHED_DURATION = 20005;
    public static final int FFP_PROP_INT64_AUDIO_CACHED_DURATION = 20006;
    public static final int FFP_PROP_INT64_VIDEO_CACHED_BYTES = 20007;
    public static final int FFP_PROP_INT64_AUDIO_CACHED_BYTES = 20008;
    public static final int FFP_PROP_INT64_VIDEO_CACHED_PACKETS = 20009;
    public static final int FFP_PROP_INT64_AUDIO_CACHED_PACKETS = 20010;
    public static final int FFP_PROP_INT64_ASYNC_STATISTIC_BUF_BACKWARDS = 20201;
    public static final int FFP_PROP_INT64_ASYNC_STATISTIC_BUF_FORWARDS = 20202;
    public static final int FFP_PROP_INT64_ASYNC_STATISTIC_BUF_CAPACITY = 20203;
    public static final int FFP_PROP_INT64_TRAFFIC_STATISTIC_BYTE_COUNT = 20204;
    public static final int FFP_PROP_INT64_CACHE_STATISTIC_PHYSICAL_POS = 20205;
    public static final int FFP_PROP_INT64_CACHE_STATISTIC_FILE_FORWARDS = 20206;
    public static final int FFP_PROP_INT64_CACHE_STATISTIC_FILE_POS = 20207;
    public static final int FFP_PROP_INT64_CACHE_STATISTIC_COUNT_BYTES = 20208;
    public static final int FFP_PROP_INT64_LOGICAL_FILE_SIZE = 20209;
    public static final int FFP_PROP_INT64_SHARE_CACHE_DATA = 20210;
    public static final int FFP_PROP_INT64_BIT_RATE = 20100;
    public static final int FFP_PROP_INT64_TCP_SPEED = 20200;
    public static final int FFP_PROP_INT64_LATEST_SEEK_LOAD_DURATION = 20300;
    public static final int FFP_PROP_INT64_IMMEDIATE_RECONNECT = 20211;
    //----------------------------------------

    @AccessedByNative
    private long mNativeMediaPlayer;
    @AccessedByNative
    private long mNativeMediaDataSource;

    @AccessedByNative
    private long mNativeAndroidIO;

    @AccessedByNative
    private int mNativeSurfaceTexture;

    @AccessedByNative
    private int mListenerContext;

    private SurfaceHolder mSurfaceHolder;
    private EventHandler mEventHandler;
    private PowerManager.WakeLock mWakeLock = null;
    private boolean mScreenOnWhilePlaying;
    private boolean mStayAwake;

    private int mVideoWidth;
    private int mVideoHeight;
    private int mVideoSarNum;
    private int mVideoSarDen;

    private String mDataSource;

    //statistic info for pressure test & troubleshooting purpose
    private long mBeginTime; //beginning time of current playback
    private long mTotalPlayTime;
    private long mLastPauseTime;
    private long mTotalPauseTime;
    private boolean bIsPaused;
    private int mDecryptMode = KANTVUtils.DECRYPT_SOFT;
    private boolean mDisableAudioTrack = false;
    private boolean mDisableVideoTrack = false;
    private boolean mEnableDumpVideoES = false;
    private boolean mEnableDumpAudioES = false;
    private int mDurationDumpES = 10; //default dumped duration is 10 seconds

    private boolean mEnableRecord = false;
    private boolean mEnableASR    = false;

    private String mVideoTitle;

    /**
     * Default library loader
     * Load them by yourself, if your libraries are not installed at default place.
     */
    private static final IjkLibLoader sLocalLibLoader = new IjkLibLoader() {
        @Override
        public void loadLibrary(String libName) throws UnsatisfiedLinkError, SecurityException {
            //System.loadLibrary(libName);
            KANTVLog.d(TAG, "load library:" + libName);
            KANTVLibraryLoader.load(libName);
            KANTVLog.d(TAG, "after load library:" + libName);
        }
    };

    private static volatile boolean mIsLibLoaded = false;

    public static void loadLibrariesOnce(IjkLibLoader libLoader) {
        synchronized (FFmpegMediaPlayer.class) {
            if (!mIsLibLoaded) {
                if (libLoader == null)
                    libLoader = sLocalLibLoader;

                //already initialized at the very beginning of app
                //since 03-25-2024 because of new software architecture
                mIsLibLoaded = true;
            }
        }
    }

    private static volatile boolean mIsNativeInitialized = false;

    private static void initNativeOnce() {
        synchronized (FFmpegMediaPlayer.class) {
            if (!mIsNativeInitialized) {
                native_init();
                mIsNativeInitialized = true;
            }
        }
    }

    /**
     * Default constructor. Consider using one of the create() methods for
     * synchronously instantiating a IjkMediaPlayer from a Uri or resource.
     * <p>
     * When done with the IjkMediaPlayer, you should call {@link #release()}, to
     * free the resources. If not released, too many IjkMediaPlayer instances
     * may result in an exception.
     * </p>
     */
    public FFmpegMediaPlayer() {
        this(sLocalLibLoader);
    }

    /**
     * do not loadLibaray
     *
     * @param libLoader custom library loader, can be null.
     */
    public FFmpegMediaPlayer(IjkLibLoader libLoader) {
        initPlayer(libLoader);
    }

    private void initPlayer(IjkLibLoader libLoader) {
        KANTVLog.d(TAG, "initPlayer");
        loadLibrariesOnce(libLoader);
        initNativeOnce();

        //should I put it in another better place?
        mBeginTime = System.currentTimeMillis();

        Looper looper;
        if ((looper = Looper.myLooper()) != null) {
            mEventHandler = new EventHandler(this, looper);
        } else if ((looper = Looper.getMainLooper()) != null) {
            mEventHandler = new EventHandler(this, looper);
        } else {
            mEventHandler = null;
        }

        /*
         * Native setup requires a weak reference to our object. It's easier to
         * create it here than in C++.
         */
        native_setup(new WeakReference<FFmpegMediaPlayer>(this));
    }

    private native void _setFrameAtTime(String imgCachePath, long startTime, long endTime, int num, int imgDefinition)
            throws IllegalArgumentException, IllegalStateException;

    /*
     * Update the IjkMediaPlayer SurfaceTexture. Call after setting a new
     * display surface.
     */
    private native void _setVideoSurface(Surface surface);

    /**
     * Sets the {@link SurfaceHolder} to use for displaying the video portion of
     * the media.
     * <p>
     * Either a surface holder or surface must be set if a display or video sink
     * is needed. Not calling this method or {@link #setSurface(Surface)} when
     * playing back a video will result in only the audio track being played. A
     * null surface holder or surface will result in only the audio track being
     * played.
     *
     * @param sh the SurfaceHolder to use for video display
     */
    @Override
    public void setDisplay(SurfaceHolder sh) {
        mSurfaceHolder = sh;
        Surface surface;
        if (sh != null) {
            surface = sh.getSurface();
        } else {
            surface = null;
        }
        _setVideoSurface(surface);
        updateSurfaceScreenOn();
    }

    /**
     * Sets the {@link Surface} to be used as the sink for the video portion of
     * the media. This is similar to {@link #setDisplay(SurfaceHolder)}, but
     * does not support {@link #setScreenOnWhilePlaying(boolean)}. Setting a
     * Surface will un-set any Surface or SurfaceHolder that was previously set.
     * A null surface will result in only the audio track being played.
     * <p>
     * If the Surface sends frames to a {@link SurfaceTexture}, the timestamps
     * returned from {@link SurfaceTexture#getTimestamp()} will have an
     * unspecified zero point. These timestamps cannot be directly compared
     * between different media sources, different instances of the same media
     * source, or multiple runs of the same program. The timestamp is normally
     * monotonically increasing and is unaffected by time-of-day adjustments,
     * but it is reset when the position is set.
     *
     * @param surface The {@link Surface} to be used for the video portion of the
     *                media.
     */
    @Override
    public void setSurface(Surface surface) {
        if (mScreenOnWhilePlaying && surface != null) {
            KANTVLog.w(TAG, "setScreenOnWhilePlaying(true) is ineffective for Surface");
        }
        mSurfaceHolder = null;
        _setVideoSurface(surface);
        updateSurfaceScreenOn();
    }

    /**
     * Sets the data source as a content Uri.
     *
     * @param context the Context to use when resolving the Uri
     * @param uri     the Content URI of the data you want to play
     * @throws IllegalStateException if it is called in an invalid state
     */
    @Override
    public void setDataSource(Context context, Uri uri)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException {
        setDataSource(context, uri, null);
    }

    /**
     * Sets the data source as a content Uri.
     *
     * @param context the Context to use when resolving the Uri
     * @param uri     the Content URI of the data you want to play
     * @param headers the headers to be sent together with the request for the data
     *                Note that the cross domain redirection is allowed by default, but that can be
     *                changed with key/value pairs through the headers parameter with
     *                "android-allow-cross-domain-redirect" as the key and "0" or "1" as the value
     *                to disallow or allow cross domain redirection.
     * @throws IllegalStateException if it is called in an invalid state
     */
    @TargetApi(Build.VERSION_CODES.ICE_CREAM_SANDWICH)
    @Override
    public void setDataSource(Context context, Uri uri, Map<String, String> headers)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException {
        final String scheme = uri.getScheme();
        if (ContentResolver.SCHEME_FILE.equals(scheme)) {
            setDataSource(uri.getPath());
            return;
        } else if (ContentResolver.SCHEME_CONTENT.equals(scheme)
                && Settings.AUTHORITY.equals(uri.getAuthority())) {
            // Redirect ringtones to go directly to underlying provider
            uri = RingtoneManager.getActualDefaultRingtoneUri(context,
                    RingtoneManager.getDefaultType(uri));
            if (uri == null) {
                throw new FileNotFoundException("Failed to resolve default ringtone");
            }
        }

        AssetFileDescriptor fd = null;
        try {
            ContentResolver resolver = context.getContentResolver();
            fd = resolver.openAssetFileDescriptor(uri, "r");
            if (fd == null) {
                return;
            }
            // Note: using getDeclaredLength so that our behavior is the same
            // as previous versions when the content provider is returning
            // a full file.
            if (fd.getDeclaredLength() < 0) {
                setDataSource(fd.getFileDescriptor());
            } else {
                setDataSource(fd.getFileDescriptor(), fd.getStartOffset(), fd.getDeclaredLength());
            }
            return;
        } catch (SecurityException ignored) {
        } catch (IOException ignored) {
        } finally {
            if (fd != null) {
                fd.close();
            }
        }

        KANTVLog.d(TAG, "uri:" + uri.toString());

        setDataSource(uri.toString(), headers);
    }

    /**
     * Sets the data source (file-path or http/rtsp URL) to use.
     *
     * @param path the path of the file, or the http/rtsp URL of the stream you
     *             want to play
     * @throws IllegalStateException if it is called in an invalid state
     *
     *                               <p>
     *                               When <code>path</code> refers to a local file, the file may
     *                               actually be opened by a process other than the calling
     *                               application. This implies that the pathname should be an
     *                               absolute path (as any other process runs with unspecified
     *                               current working directory), and that the pathname should
     *                               reference a world-readable file.
     */
    @Override
    public void setDataSource(String path)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException {
        mDataSource = path;
        _setDataSource(path, null, null);
    }

    /**
     * Sets the data source (file-path or http/rtsp URL) to use.
     *
     * @param path    the path of the file, or the http/rtsp URL of the stream you want to play
     * @param headers the headers associated with the http request for the stream you want to play
     * @throws IllegalStateException if it is called in an invalid state
     */
    public void setDataSource(String path, Map<String, String> headers)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException {
        if (headers != null && !headers.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                sb.append(entry.getKey());
                sb.append(":");
                String value = entry.getValue();
                if (!TextUtils.isEmpty(value))
                    sb.append(entry.getValue());
                sb.append("\r\n");
                setOption(OPT_CATEGORY_FORMAT, "headers", sb.toString());
                setOption(FFmpegMediaPlayer.OPT_CATEGORY_FORMAT, "protocol_whitelist", "async,cache,crypto,file,http,https,kantvhttphook,kantvinject,kantvlivehook,kantvlongurl,kantvsegment,kantvtcphook,pipe,rtp,tcp,tls,udp,kantvurlhook,data");
            }
        }
        setDataSource(path);
    }

    /**
     * Sets the data source (FileDescriptor) to use. It is the caller's responsibility
     * to close the file descriptor. It is safe to do so as soon as this call returns.
     *
     * @param fd the FileDescriptor for the file you want to play
     * @throws IllegalStateException if it is called in an invalid state
     */
    @TargetApi(Build.VERSION_CODES.HONEYCOMB_MR2)
    @Override
    public void setDataSource(FileDescriptor fd)
            throws IOException, IllegalArgumentException, IllegalStateException {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.HONEYCOMB_MR1) {
            int native_fd = -1;
            try {
                Field f = fd.getClass().getDeclaredField("descriptor"); //NoSuchFieldException
                f.setAccessible(true);
                native_fd = f.getInt(fd); //IllegalAccessException
            } catch (NoSuchFieldException e) {
                throw new RuntimeException(e);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            }
            _setDataSourceFd(native_fd);
        } else {
            ParcelFileDescriptor pfd = ParcelFileDescriptor.dup(fd);
            try {
                _setDataSourceFd(pfd.getFd());
            } finally {
                pfd.close();
            }
        }
    }

    /**
     * Sets the data source (FileDescriptor) to use.  The FileDescriptor must be
     * seekable (N.B. a LocalSocket is not seekable). It is the caller's responsibility
     * to close the file descriptor. It is safe to do so as soon as this call returns.
     *
     * @param fd     the FileDescriptor for the file you want to play
     * @param offset the offset into the file where the data to be played starts, in bytes
     * @param length the length in bytes of the data to be played
     * @throws IllegalStateException if it is called in an invalid state
     */
    private void setDataSource(FileDescriptor fd, long offset, long length)
            throws IOException, IllegalArgumentException, IllegalStateException {
        // FIXME: handle offset, length
        setDataSource(fd);
    }

    public void setDataSource(IMediaDataSource mediaDataSource)
            throws IllegalArgumentException, SecurityException, IllegalStateException {
        _setDataSource(mediaDataSource);
    }

    public void setAndroidIOCallback(IAndroidIO androidIO)
            throws IllegalArgumentException, SecurityException, IllegalStateException {
        _setAndroidIOCallback(androidIO);
    }

    private native void _setDataSource(String path, String[] keys, String[] values)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException;

    private native void _setDataSourceFd(int fd)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException;

    private native void _setDataSource(IMediaDataSource mediaDataSource)
            throws IllegalArgumentException, SecurityException, IllegalStateException;

    private native void _setAndroidIOCallback(IAndroidIO androidIO)
            throws IllegalArgumentException, SecurityException, IllegalStateException;

    @Override
    public String getDataSource() {
        return mDataSource;
    }


    @Override
    public void setVideoTitle(String title) {
        mVideoTitle = title;
    }

    @Override
    public String getVideoTitle() {
        return mVideoTitle;
    }

    @Override
    public void prepareAsync() throws IllegalStateException {
        _prepareAsync();
    }

    public native void _prepareAsync() throws IllegalStateException;

    @Override
    public void start() throws IllegalStateException {
        KANTVLog.i(TAG, "start playing...");
        stayAwake(true);
        if (bIsPaused) {
            mTotalPauseTime += System.currentTimeMillis() - mLastPauseTime;
            Log.i(TAG, "mTotalPauseTime " + (mTotalPauseTime / 1000));
        }

        ITrackInfo trackInfos[] = getTrackInfo();
        if (trackInfos != null) {
            int index = -1;
            for (ITrackInfo trackInfo : trackInfos) {
                index++;
                int trackType = trackInfo.getTrackType();
                IMediaFormat mediaFormat = trackInfo.getFormat();
                if (mediaFormat != null) {
                    switch (trackType) {
                        case ITrackInfo.MEDIA_TRACK_TYPE_VIDEO:
                            if (mDisableVideoTrack) {
                                KANTVLog.d(TAG, "disable video track");
                                deselectTrack(index);
                            }
                            break;
                        case ITrackInfo.MEDIA_TRACK_TYPE_AUDIO:
                            if (mDisableAudioTrack) {
                                KANTVLog.d(TAG, "disable audio track");
                                deselectTrack(index);
                            }
                            break;
                        case ITrackInfo.MEDIA_TRACK_TYPE_SUBTITLE:
                            KANTVLog.d(TAG, "found subtitle track");
                            break;
                        case ITrackInfo.MEDIA_TRACK_TYPE_METADATA:
                            KANTVLog.d(TAG, "found metadata");
                            break;
                        case ITrackInfo.MEDIA_TRACK_TYPE_TIMEDTEXT:
                            KANTVLog.d(TAG, "found timedtext");
                        default:
                            break;
                    }
                }
            }
        }

        _setEnableDumpES(KANTVUtils.VIDEO_TRACK, mEnableDumpVideoES ? 1 : 0);
        _setEnableDumpES(KANTVUtils.AUDIO_TRACK, mEnableDumpAudioES ? 1 : 0);
        _setDurationDumpES(mDurationDumpES);

        _start();
        bIsPaused = false;
    }

    private native void _start() throws IllegalStateException;

    @Override
    public void stop() throws IllegalStateException {
        KANTVLog.i(TAG, "stop playing...");
        stayAwake(false);
        _stop();
    }

    private native void _stop() throws IllegalStateException;

    @Override
    public void pause() throws IllegalStateException {
        stayAwake(false);
        assert (bIsPaused == false);

        if (!bIsPaused) {
            mLastPauseTime = System.currentTimeMillis();
        }
        KANTVLog.i(TAG, "pause playing...");
        _pause();
        bIsPaused = !bIsPaused;
    }

    private native void _pause() throws IllegalStateException;


    public long getBeginTime() {
        return mBeginTime;
    }


    public long getPauseTime() {
        long mCurrentPauseTime = 0;
        if (bIsPaused) {
            mCurrentPauseTime = System.currentTimeMillis() - mLastPauseTime;
        }
        return (mTotalPauseTime + mCurrentPauseTime) / 1000;
    }


    public long getPlayTime() {
        DateFormat dateFormatterChina = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.MEDIUM);
        TimeZone timeZoneChina = TimeZone.getTimeZone("Asia/Shanghai");
        dateFormatterChina.setTimeZone(timeZoneChina);

        if (isPlaying()) {
            long curTime = System.currentTimeMillis();
            mTotalPlayTime = ((curTime - mBeginTime - mTotalPauseTime) / 1000);

            return mTotalPlayTime;
        }
        return mTotalPlayTime;
    }


    @SuppressLint("Wakelock")
    @Override
    public void setWakeMode(Context context, int mode) {
        boolean washeld = false;
        if (mWakeLock != null) {
            if (mWakeLock.isHeld()) {
                washeld = true;
                mWakeLock.release();
            }
            mWakeLock = null;
        }

        PowerManager pm = (PowerManager) context
                .getSystemService(Context.POWER_SERVICE);
        mWakeLock = pm.newWakeLock(mode | PowerManager.ON_AFTER_RELEASE,
                FFmpegMediaPlayer.class.getName());
        mWakeLock.setReferenceCounted(false);
        if (washeld) {
            mWakeLock.acquire();
        }
    }

    @Override
    public void setScreenOnWhilePlaying(boolean screenOn) {
        if (mScreenOnWhilePlaying != screenOn) {
            if (screenOn && mSurfaceHolder == null) {
                KANTVLog.w(TAG, "setScreenOnWhilePlaying(true) is ineffective without a SurfaceHolder");
            }
            mScreenOnWhilePlaying = screenOn;
            updateSurfaceScreenOn();
        }
    }

    @SuppressLint("Wakelock")
    private void stayAwake(boolean awake) {
        if (mWakeLock != null) {
            if (awake && !mWakeLock.isHeld()) {
                mWakeLock.acquire();
            } else if (!awake && mWakeLock.isHeld()) {
                mWakeLock.release();
            }
        }
        mStayAwake = awake;
        updateSurfaceScreenOn();
    }

    private void updateSurfaceScreenOn() {
        if (mSurfaceHolder != null) {
            mSurfaceHolder.setKeepScreenOn(mScreenOnWhilePlaying && mStayAwake);
        }
    }

    @Override
    public IjkTrackInfo[] getTrackInfo() {
        Bundle bundle = getMediaMeta();
        if (bundle == null)
            return null;

        IjkMediaMeta mediaMeta = IjkMediaMeta.parse(bundle);
        if (mediaMeta == null || mediaMeta.mStreams == null)
            return null;

        ArrayList<IjkTrackInfo> trackInfos = new ArrayList<IjkTrackInfo>();
        for (IjkMediaMeta.IjkStreamMeta streamMeta : mediaMeta.mStreams) {
            IjkTrackInfo trackInfo = new IjkTrackInfo(streamMeta);
            if (streamMeta.mType.equalsIgnoreCase(IjkMediaMeta.IJKM_VAL_TYPE__VIDEO)) {
                trackInfo.setTrackType(ITrackInfo.MEDIA_TRACK_TYPE_VIDEO);
            } else if (streamMeta.mType.equalsIgnoreCase(IjkMediaMeta.IJKM_VAL_TYPE__AUDIO)) {
                trackInfo.setTrackType(ITrackInfo.MEDIA_TRACK_TYPE_AUDIO);
            } else if (streamMeta.mType.equalsIgnoreCase(IjkMediaMeta.IJKM_VAL_TYPE__TIMEDTEXT)) {
                trackInfo.setTrackType(ITrackInfo.MEDIA_TRACK_TYPE_TIMEDTEXT);
            }
            trackInfos.add(trackInfo);
        }

        return trackInfos.toArray(new IjkTrackInfo[trackInfos.size()]);
    }

    // TODO: @Override
    public int getSelectedTrack(int trackType) {
        switch (trackType) {
            case ITrackInfo.MEDIA_TRACK_TYPE_VIDEO:
                return (int) _getPropertyLong(FFP_PROP_INT64_SELECTED_VIDEO_STREAM, -1);
            case ITrackInfo.MEDIA_TRACK_TYPE_AUDIO:
                return (int) _getPropertyLong(FFP_PROP_INT64_SELECTED_AUDIO_STREAM, -1);
            case ITrackInfo.MEDIA_TRACK_TYPE_TIMEDTEXT:
                return (int) _getPropertyLong(FFP_PROP_INT64_SELECTED_TIMEDTEXT_STREAM, -1);
            default:
                return -1;
        }
    }

    public void setTFLite(boolean enableTFLite) {
        _setTFLite(enableTFLite);
    }

    // experimental, should set DEFAULT_MIN_FRAMES and MAX_MIN_FRAMES to 25
    // TODO: @Override
    public void selectTrack(int track) {
        _setStreamSelected(track, true);
    }

    // experimental, should set DEFAULT_MIN_FRAMES and MAX_MIN_FRAMES to 25
    // TODO: @Override
    public void deselectTrack(int track) {
        _setStreamSelected(track, false);
    }

    private native void _setTFLite(boolean enabled);

    private native void _setStreamSelected(int stream, boolean select);

    @Override
    public int getVideoWidth() {
        return mVideoWidth;
    }

    @Override
    public int getVideoHeight() {
        return mVideoHeight;
    }

    @Override
    public int getVideoSarNum() {
        return mVideoSarNum;
    }

    @Override
    public int getVideoSarDen() {
        return mVideoSarDen;
    }

    @Override
    public native boolean isPlaying();

    @Override
    public native boolean isClearContent();

    //TODO: not implemented
    @Override
    public void setDecryptMode(int decryptMode) {
        //TODO: TEE decrypt via FFmpeg
        mDecryptMode = decryptMode;
    }

    public native int getDrmSystemId();


    @Override
    public String getDrmInfo() {
        int drmSystemId = getDrmSystemId();
        return "ChinaDRM";
    }

    //TODO: not implemented
    @Override
    public void selectTrack(TrackInfoBean trackInfo, boolean isAudio) {
    }


    @Override
    public void setDisableAudioTrack(boolean bDisable) {
        mDisableAudioTrack = bDisable;
    }


    @Override
    public void setDisableVideoTrack(boolean bDisable) {
        mDisableVideoTrack = bDisable;
    }

    @Override
    public void setEnableDumpVideoES(boolean bEnableDumpVideoES) {
        mEnableDumpVideoES = bEnableDumpVideoES;
        KANTVLog.d(TAG, "enable dump video es: " + mEnableDumpVideoES);
        _setEnableDumpES(KANTVUtils.VIDEO_TRACK, mEnableDumpVideoES ? 1 : 0);
    }

    @Override
    public void setEnableDumpAudioES(boolean bEnableDumpAudioES) {
        mEnableDumpAudioES = bEnableDumpAudioES;
        KANTVLog.d(TAG, "enable dump audio es: " + bEnableDumpAudioES);
        _setEnableDumpES(KANTVUtils.AUDIO_TRACK, mEnableDumpAudioES ? 1 : 0);
    }

    @Override
    public void setEnableRecord(boolean bEnableRecord) {
        mEnableRecord = bEnableRecord;
        KANTVLog.d(TAG, "enable record: " + mEnableRecord);
        _setEnableRecord(mEnableRecord ? 1 : 0);
    }


    //TODO
    @Override
    public void updateRecordingStatus(boolean bEnableRecord) {
    }

    @Override
    public boolean getEnableRecord() {
        int isRecording = _getEnableRecord();
        if (0 == isRecording)
            mEnableRecord = false;
        else
            mEnableRecord = true;
        return mEnableRecord;
    }

    @Override
    public void setEnableASR(boolean bEnableASR) {
        mEnableASR = bEnableASR;
        KANTVLog.d(TAG, "enable ASR: " + mEnableASR);
        _setEnableASR(mEnableASR ? 1 : 0);
    }

    @Override
    public boolean getEnableASR() {
        int isASRing= _getEnableASR();
        if (0 == isASRing)
            mEnableASR = false;
        else
            mEnableASR = true;
        return mEnableASR;
    }


    @Override
    public void setDurationDumpES(int durationDumpES) {
        KANTVLog.d(TAG, "duration is : " + durationDumpES);
        mDurationDumpES = durationDumpES;
        _setDurationDumpES(mDurationDumpES);
    }

    public native String getVideoInfo();

    public native String getAudioInfo();

    public native String getFFmpegVersion();

    public native String getBenchmarkInfo();
    public native void   startBenchmark();

    @Override
    public String getVideoDecoderName() {
        return "FFmpeg";
    }

    @Override
    public String getAudioDecoderName() {
        return "FFmpeg";
    }

    /**
     * @param streamType 0: video 1: audio
     * @return 1: enable dump es  0: disable dump es
     */
    public native int _getEnableDumpES(int streamType);

    /**
     * @param streamType    0: video 1: audio
     * @param bEnableDumpES true: enable dump es  false: disable dump es
     */
    public native void _setEnableDumpES(int streamType, int bEnableDumpES);

    //default dumped video/audio duration is 10 seconds
    public native void _setDurationDumpES(int dumpESDuration);
    //end added

    public native void _setEnableRecord(int bEnableRecord);

    public native int _getEnableRecord();

    public native void _setEnableASR(int bEnableASR);

    public native int _getEnableASR();

    @Override
    public native void seekTo(long msec) throws IllegalStateException;

    @Override
    public native long getCurrentPosition();

    @Override
    public native long getDuration();

    /**
     * Releases resources associated with this IjkMediaPlayer object. It is
     * considered good practice to call this method when you're done using the
     * IjkMediaPlayer. In particular, whenever an Activity of an application is
     * paused (its onPause() method is called), or stopped (its onStop() method
     * is called), this method should be invoked to release the IjkMediaPlayer
     * object, unless the application has a special need to keep the object
     * around. In addition to unnecessary resources (such as memory and
     * instances of codecs) being held, failure to call this method immediately
     * if a IjkMediaPlayer object is no longer needed may also lead to
     * continuous battery consumption for mobile devices, and playback failure
     * for other applications if no multiple instances of the same codec are
     * supported on a device. Even if multiple instances of the same codec are
     * supported, some performance degradation may be expected when unnecessary
     * multiple instances are used at the same time.
     */
    @Override
    public void release() {
        KANTVLog.i(TAG, "release");
        stayAwake(false);
        updateSurfaceScreenOn();
        resetListeners();
        _release();
    }

    private native void _release();

    @Override
    public void reset() {
        KANTVLog.i(TAG, "reset");
        stayAwake(false);
        _reset();
        // make sure none of the listeners get called anymore
        mEventHandler.removeCallbacksAndMessages(null);

        mVideoWidth = 0;
        mVideoHeight = 0;
    }

    private native void _reset();

    /**
     * Sets the player to be looping or non-looping.
     *
     * @param looping whether to loop or not
     */
    @Override
    public void setLooping(boolean looping) {
        int loopCount = looping ? 0 : 1;
        setOption(OPT_CATEGORY_PLAYER, "loop", loopCount);
        _setLoopCount(loopCount);
    }

    private native void _setLoopCount(int loopCount);

    /**
     * Checks whether the MediaPlayer is looping or non-looping.
     *
     * @return true if the MediaPlayer is currently looping, false otherwise
     */
    @Override
    public boolean isLooping() {
        int loopCount = _getLoopCount();
        return loopCount != 1;
    }

    private native int _getLoopCount();

    public void setSpeed(float speed) {
        _setPropertyFloat(FFP_PROP_FLOAT_PLAYBACK_RATE, speed);
    }

    public float getSpeed() {
        return _getPropertyFloat(FFP_PROP_FLOAT_PLAYBACK_RATE, 1.0f);
    }

    public void setPlaybackSpeed(float speed) {
        _setPropertyFloat(FFP_PROP_FLOAT_PLAYBACK_RATE, speed);
    }

    public float getPlaybackSpeed() {
        return _getPropertyFloat(FFP_PROP_FLOAT_PLAYBACK_RATE, .0f);
    }

    public int getVideoDecoder() {
        return (int) _getPropertyLong(FFP_PROP_INT64_VIDEO_DECODER, FFP_PROPV_DECODER_UNKNOWN);
    }

    public float getVideoOutputFramesPerSecond() {
        return _getPropertyFloat(PROP_FLOAT_VIDEO_OUTPUT_FRAMES_PER_SECOND, 0.0f);
    }

    public float getVideoDecodeFramesPerSecond() {
        return _getPropertyFloat(PROP_FLOAT_VIDEO_DECODE_FRAMES_PER_SECOND, 0.0f);
    }

    public long getVideoCachedDuration() {
        return _getPropertyLong(FFP_PROP_INT64_VIDEO_CACHED_DURATION, 0);
    }

    public long getAudioCachedDuration() {
        return _getPropertyLong(FFP_PROP_INT64_AUDIO_CACHED_DURATION, 0);
    }

    public long getVideoCachedBytes() {
        return _getPropertyLong(FFP_PROP_INT64_VIDEO_CACHED_BYTES, 0);
    }

    public long getAudioCachedBytes() {
        return _getPropertyLong(FFP_PROP_INT64_AUDIO_CACHED_BYTES, 0);
    }

    public long getVideoCachedPackets() {
        return _getPropertyLong(FFP_PROP_INT64_VIDEO_CACHED_PACKETS, 0);
    }

    public long getAudioCachedPackets() {
        return _getPropertyLong(FFP_PROP_INT64_AUDIO_CACHED_PACKETS, 0);
    }

    public long getAsyncStatisticBufBackwards() {
        return _getPropertyLong(FFP_PROP_INT64_ASYNC_STATISTIC_BUF_BACKWARDS, 0);
    }

    public long getAsyncStatisticBufForwards() {
        return _getPropertyLong(FFP_PROP_INT64_ASYNC_STATISTIC_BUF_FORWARDS, 0);
    }

    public long getAsyncStatisticBufCapacity() {
        return _getPropertyLong(FFP_PROP_INT64_ASYNC_STATISTIC_BUF_CAPACITY, 0);
    }

    public long getTrafficStatisticByteCount() {
        return _getPropertyLong(FFP_PROP_INT64_TRAFFIC_STATISTIC_BYTE_COUNT, 0);
    }

    public long getCacheStatisticPhysicalPos() {
        return _getPropertyLong(FFP_PROP_INT64_CACHE_STATISTIC_PHYSICAL_POS, 0);
    }

    public long getCacheStatisticFileForwards() {
        return _getPropertyLong(FFP_PROP_INT64_CACHE_STATISTIC_FILE_FORWARDS, 0);
    }

    public long getCacheStatisticFilePos() {
        return _getPropertyLong(FFP_PROP_INT64_CACHE_STATISTIC_FILE_POS, 0);
    }

    public long getCacheStatisticCountBytes() {
        return _getPropertyLong(FFP_PROP_INT64_CACHE_STATISTIC_COUNT_BYTES, 0);
    }

    public long getFileSize() {
        return _getPropertyLong(FFP_PROP_INT64_LOGICAL_FILE_SIZE, 0);
    }

    @Override
    public long getBitRate() {
        return _getPropertyLong(FFP_PROP_INT64_BIT_RATE, 0);
    }

    public long getTcpSpeed() {
        return _getPropertyLong(FFP_PROP_INT64_TCP_SPEED, 0);
    }

    public long getSeekLoadDuration() {
        return _getPropertyLong(FFP_PROP_INT64_LATEST_SEEK_LOAD_DURATION, 0);
    }


    private native float _getPropertyFloat(int property, float defaultValue);

    private native void _setPropertyFloat(int property, float value);

    private native long _getPropertyLong(int property, long defaultValue);

    private native void _setPropertyLong(int property, long value);

    public float getDropFrameRate() {
        return _getPropertyFloat(FFP_PROP_FLOAT_DROP_FRAME_RATE, .0f);
    }

    @Override
    public native void setVolume(float leftVolume, float rightVolume);

    @Override
    public native int getAudioSessionId();

    @Override
    public IjkMediaInfo getMediaInfo() {
        IjkMediaInfo mediaInfo = new IjkMediaInfo();
        mediaInfo.mMediaPlayerName = "ffmpeg";

        String videoCodecInfo = _getVideoCodecInfo();
        if (!TextUtils.isEmpty(videoCodecInfo)) {
            String nodes[] = videoCodecInfo.split(",");
            if (nodes.length >= 2) {
                mediaInfo.mVideoDecoder = nodes[0];
                mediaInfo.mVideoDecoderImpl = nodes[1];
            } else if (nodes.length >= 1) {
                mediaInfo.mVideoDecoder = nodes[0];
                mediaInfo.mVideoDecoderImpl = "";
            }
        }

        String audioCodecInfo = _getAudioCodecInfo();
        if (!TextUtils.isEmpty(audioCodecInfo)) {
            String nodes[] = audioCodecInfo.split(",");
            if (nodes.length >= 2) {
                mediaInfo.mAudioDecoder = nodes[0];
                mediaInfo.mAudioDecoderImpl = nodes[1];
            } else if (nodes.length >= 1) {
                mediaInfo.mAudioDecoder = nodes[0];
                mediaInfo.mAudioDecoderImpl = "";
            }
        }

        try {
            mediaInfo.mMeta = IjkMediaMeta.parse(_getMediaMeta());
        } catch (Throwable e) {
            e.printStackTrace();
        }
        return mediaInfo;
    }

    @Override
    public void setLogEnabled(boolean enable) {
        // do nothing
    }

    @Override
    public boolean isPlayable() {
        return true;
    }

    private native String _getVideoCodecInfo();

    private native String _getAudioCodecInfo();

    public void setOption(int category, String name, String value) {
        _setOption(category, name, value);
    }

    public void setOption(int category, String name, long value) {
        _setOption(category, name, value);
    }

    private native void _setOption(int category, String name, String value);

    private native void _setOption(int category, String name, long value);

    public Bundle getMediaMeta() {
        return _getMediaMeta();
    }

    private native Bundle _getMediaMeta();

    public static String getColorFormatName(int mediaCodecColorFormat) {
        return _getColorFormatName(mediaCodecColorFormat);
    }

    private static native String _getColorFormatName(int mediaCodecColorFormat);

    @Override
    public void setAudioStreamType(int streamtype) {
        // do nothing
    }

    @Override
    public void setKeepInBackground(boolean keepInBackground) {
        // do nothing
    }

    private static native void native_init();

    private native void native_setup(Object IjkMediaPlayer_this);

    private native void native_finalize();

    private native void native_message_loop(Object IjkMediaPlayer_this);

    protected void finalize() throws Throwable {
        super.finalize();
        native_finalize();
    }

    public void httphookReconnect() {
        _setPropertyLong(FFP_PROP_INT64_IMMEDIATE_RECONNECT, 1);
    }

    public void setCacheShare(int share) {
        _setPropertyLong(FFP_PROP_INT64_SHARE_CACHE_DATA, (long) share);
    }

    private static class EventHandler extends Handler {
        private final WeakReference<FFmpegMediaPlayer> mWeakPlayer;

        public EventHandler(FFmpegMediaPlayer mp, Looper looper) {
            super(looper);
            mWeakPlayer = new WeakReference<FFmpegMediaPlayer>(mp);
        }

        @Override
        public void handleMessage(Message msg) {
            FFmpegMediaPlayer player = mWeakPlayer.get();
            if (player == null || player.mNativeMediaPlayer == 0) {
                KANTVLog.w(TAG, "it shouldn't happen, pl check");
                return;
            }

            switch (msg.what) {
                case MEDIA_PREPARED:
                    player.notifyOnPrepared();
                    return;

                case MEDIA_PLAYBACK_COMPLETE:
                    player.stayAwake(false);
                    player.notifyOnCompletion();
                    return;

                case MEDIA_BUFFERING_UPDATE:
                    long bufferPosition = msg.arg1;
                    if (bufferPosition < 0) {
                        bufferPosition = 0;
                    }

                    long percent = 0;
                    long duration = player.getDuration();
                    if (duration > 0) {
                        percent = bufferPosition * 100 / duration;
                    }
                    if (percent >= 100) {
                        percent = 100;
                    }

                    KANTVLog.d(TAG, "Buffer: percent:" + percent + ", bufferPostion: " + bufferPosition + ", duration : " + duration);
                    player.notifyOnBufferingUpdate((int) percent);
                    return;

                case MEDIA_SEEK_COMPLETE:
                    player.notifyOnSeekComplete();
                    return;

                case MEDIA_SET_VIDEO_SIZE:
                    player.mVideoWidth = msg.arg1;
                    player.mVideoHeight = msg.arg2;
                    player.notifyOnVideoSizeChanged(player.mVideoWidth, player.mVideoHeight,
                            player.mVideoSarNum, player.mVideoSarDen);
                    return;

                case MEDIA_ERROR:
                    KANTVLog.d(TAG, "Error (" + msg.arg1 + "," + msg.arg2 + ")");
                    if (!player.notifyOnError(msg.arg1, msg.arg2)) {
                        player.notifyOnCompletion();
                    }
                    player.stayAwake(false);
                    return;

                case MEDIA_INFO:
                    switch (msg.arg1) {
                        case MEDIA_INFO_VIDEO_RENDERING_START:
                            KANTVLog.d(TAG, "Info: MEDIA_INFO_VIDEO_RENDERING_START\n");
                            break;
                    }
                    KANTVLog.d(TAG, "args1:" + msg.arg1);
                    player.notifyOnInfo(msg.arg1, msg.arg2);
                    // No real default action so far.
                    return;
                case MEDIA_TIMED_TEXT:
                    if (msg.obj == null) {
                        player.notifyOnTimedText(null);
                    } else {
                        IjkTimedText text = new IjkTimedText(new Rect(0, 0, 1, 1), (String) msg.obj);
                        player.notifyOnTimedText(text);
                    }
                    return;
                case MEDIA_NOP: // interface test message - ignore
                    break;

                case MEDIA_SET_VIDEO_SAR:
                    player.mVideoSarNum = msg.arg1;
                    player.mVideoSarDen = msg.arg2;
                    player.notifyOnVideoSizeChanged(player.mVideoWidth, player.mVideoHeight,
                            player.mVideoSarNum, player.mVideoSarDen);
                    break;
                case MEDIA_INFO_RECORDING_START:
                    KANTVUtils.setRecordingFileName((String)msg.obj);
                    player.notifyOnInfo(MEDIA_INFO_RECORDING_START,0);
                    break;
                case MEDIA_INFO_ASR_START:
                    KANTVUtils.setASRSavedFileName((String)msg.obj);
                    player.notifyOnInfo(MEDIA_INFO_ASR_START,0);
                    break;
                case MEDIA_INFO_ASR_STOP:
                    player.notifyOnInfo(MEDIA_INFO_ASR_STOP,0);
                    break;
                case MEDIA_INFO_ASR_START_NORMAL:
                    player.notifyOnInfo(MEDIA_INFO_ASR_START_NORMAL,0);
                    break;
                case MEDIA_INFO_ASR_STOP_NORMAL:
                    player.notifyOnInfo(MEDIA_INFO_ASR_STOP_NORMAL,0);
                    break;
                default:
                    KANTVLog.d(TAG, "Unknown message type " + msg.what);
            }
        }
    }

    /*
     * Called from native code when an interesting event happens. This method
     * just uses the EventHandler system to post the event back to the main app
     * thread. We use a weak reference to the original IjkMediaPlayer object so
     * that the native code is safe from the object disappearing from underneath
     * it. (This is the cookie passed to native_setup().)
     */
    @CalledByNative
    private static void postEventFromNative(Object weakThiz, int what,
                                            int arg1, int arg2, Object obj) {
        if (weakThiz == null) {
            KANTVLog.g(TAG, "why weakThiz is null");
            return;
        }

        @SuppressWarnings("rawtypes")
        FFmpegMediaPlayer mp = (FFmpegMediaPlayer) ((WeakReference) weakThiz).get();
        if (mp == null) {
            KANTVLog.g(TAG, "why mp is null");
            return;
        }

        if (what == MEDIA_INFO && arg1 == MEDIA_INFO_STARTED_AS_NEXT) {
            // this acquires the wakelock if needed, and sets the client side
            // state
            mp.start();
        }
        if (mp.mEventHandler != null) {
            Message m = mp.mEventHandler.obtainMessage(what, arg1, arg2, obj);
            mp.mEventHandler.sendMessage(m);
        } else {
            KANTVLog.g(TAG, "mp.mEventHandler is null");
        }
    }

    /*
     * ControlMessage
     */

    private OnControlMessageListener mOnControlMessageListener;

    public void setOnControlMessageListener(OnControlMessageListener listener) {
        mOnControlMessageListener = listener;
    }

    public interface OnControlMessageListener {
        String onControlResolveSegmentUrl(int segment);
    }

    /*
     * NativeInvoke
     */

    private OnNativeInvokeListener mOnNativeInvokeListener;

    public void setOnNativeInvokeListener(OnNativeInvokeListener listener) {
        mOnNativeInvokeListener = listener;
    }

    public interface OnNativeInvokeListener {

        int CTRL_WILL_TCP_OPEN = 0x20001;               // NO ARGS
        int CTRL_DID_TCP_OPEN = 0x20002;                // ARG_ERROR, ARG_FAMILIY, ARG_IP, ARG_PORT, ARG_FD

        int CTRL_WILL_HTTP_OPEN = 0x20003;              // ARG_URL, ARG_SEGMENT_INDEX, ARG_RETRY_COUNTER
        int CTRL_WILL_LIVE_OPEN = 0x20005;              // ARG_URL, ARG_RETRY_COUNTER
        int CTRL_WILL_CONCAT_RESOLVE_SEGMENT = 0x20007; // ARG_URL, ARG_SEGMENT_INDEX, ARG_RETRY_COUNTER

        int EVENT_WILL_HTTP_OPEN = 0x1;                 // ARG_URL
        int EVENT_DID_HTTP_OPEN = 0x2;                  // ARG_URL, ARG_ERROR, ARG_HTTP_CODE
        int EVENT_WILL_HTTP_SEEK = 0x3;                 // ARG_URL, ARG_OFFSET
        int EVENT_DID_HTTP_SEEK = 0x4;                  // ARG_URL, ARG_OFFSET, ARG_ERROR, ARG_HTTP_CODE, ARG_FILE_SIZE

        String ARG_URL = "url";
        String ARG_SEGMENT_INDEX = "segment_index";
        String ARG_RETRY_COUNTER = "retry_counter";

        String ARG_ERROR = "error";
        String ARG_FAMILIY = "family";
        String ARG_IP = "ip";
        String ARG_PORT = "port";
        String ARG_FD = "fd";

        String ARG_OFFSET = "offset";
        String ARG_HTTP_CODE = "http_code";
        String ARG_FILE_SIZE = "file_size";

        /*
         * @return true if invoke is handled
         * @throws Exception on any error
         */
        boolean onNativeInvoke(int what, Bundle args);
    }

    @CalledByNative
    private static boolean onNativeInvoke(Object weakThiz, int what, Bundle args) {
        KANTVLog.d(TAG, "onNativeInvoke: " + what );
        if (weakThiz == null || !(weakThiz instanceof WeakReference<?>))
            throw new IllegalStateException("<null weakThiz>.onNativeInvoke()");

        @SuppressWarnings("unchecked")
        WeakReference<FFmpegMediaPlayer> weakPlayer = (WeakReference<FFmpegMediaPlayer>) weakThiz;
        FFmpegMediaPlayer player = weakPlayer.get();
        if (player == null)
            throw new IllegalStateException("<null weakPlayer>.onNativeInvoke()");

        OnNativeInvokeListener listener = player.mOnNativeInvokeListener;
        if (listener != null && listener.onNativeInvoke(what, args))
            return true;

        switch (what) {
            case OnNativeInvokeListener.CTRL_WILL_CONCAT_RESOLVE_SEGMENT: {
                OnControlMessageListener onControlMessageListener = player.mOnControlMessageListener;
                if (onControlMessageListener == null)
                    return false;

                int segmentIndex = args.getInt(OnNativeInvokeListener.ARG_SEGMENT_INDEX, -1);
                if (segmentIndex < 0)
                    throw new InvalidParameterException("onNativeInvoke(invalid segment index)");

                String newUrl = onControlMessageListener.onControlResolveSegmentUrl(segmentIndex);
                if (newUrl == null)
                    throw new RuntimeException(new IOException("onNativeInvoke() = <NULL newUrl>"));

                args.putString(OnNativeInvokeListener.ARG_URL, newUrl);
                return true;
            }
            default:
                return false;
        }
    }

    /*
     * MediaCodec select
     */

    public interface OnMediaCodecSelectListener {
        String onMediaCodecSelect(IMediaPlayer mp, String mimeType, int profile, int level);
    }

    private OnMediaCodecSelectListener mOnMediaCodecSelectListener;

    public void setOnMediaCodecSelectListener(OnMediaCodecSelectListener listener) {
        mOnMediaCodecSelectListener = listener;
    }

    public void resetListeners() {
        super.resetListeners();
        mOnMediaCodecSelectListener = null;
    }

    @CalledByNative
    private static String onSelectCodec(Object weakThiz, String mimeType, int profile, int level) {
        if (weakThiz == null || !(weakThiz instanceof WeakReference<?>))
            return null;

        @SuppressWarnings("unchecked")
        WeakReference<FFmpegMediaPlayer> weakPlayer = (WeakReference<FFmpegMediaPlayer>) weakThiz;
        FFmpegMediaPlayer player = weakPlayer.get();
        if (player == null)
            return null;

        OnMediaCodecSelectListener listener = player.mOnMediaCodecSelectListener;
        if (listener == null)
            listener = DefaultMediaCodecSelector.sInstance;

        return listener.onMediaCodecSelect(player, mimeType, profile, level);
    }

    public static class DefaultMediaCodecSelector implements OnMediaCodecSelectListener {
        public static final DefaultMediaCodecSelector sInstance = new DefaultMediaCodecSelector();

        @SuppressWarnings("deprecation")
        @TargetApi(Build.VERSION_CODES.JELLY_BEAN)
        public String onMediaCodecSelect(IMediaPlayer mp, String mimeType, int profile, int level) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN)
                return null;

            if (TextUtils.isEmpty(mimeType))
                return null;

            Log.i(TAG, String.format(Locale.US, "onSelectCodec: mime=%s, profile=%d, level=%d", mimeType, profile, level));
            ArrayList<IjkMediaCodecInfo> candidateCodecList = new ArrayList<IjkMediaCodecInfo>();
            int numCodecs = MediaCodecList.getCodecCount();
            for (int i = 0; i < numCodecs; i++) {
                MediaCodecInfo codecInfo = MediaCodecList.getCodecInfoAt(i);
                Log.d(TAG, String.format(Locale.US, "  found codec: %s", codecInfo.getName()));
                if (codecInfo.isEncoder())
                    continue;

                String[] types = codecInfo.getSupportedTypes();
                if (types == null)
                    continue;

                for (String type : types) {
                    if (TextUtils.isEmpty(type))
                        continue;

                    Log.d(TAG, String.format(Locale.US, "    mime: %s", type));
                    if (!type.equalsIgnoreCase(mimeType))
                        continue;

                    IjkMediaCodecInfo candidate = IjkMediaCodecInfo.setupCandidate(codecInfo, mimeType);
                    if (candidate == null)
                        continue;

                    candidateCodecList.add(candidate);
                    Log.i(TAG, String.format(Locale.US, "candidate codec: %s rank=%d", codecInfo.getName(), candidate.mRank));
                    candidate.dumpProfileLevels(mimeType);
                }
            }

            if (candidateCodecList.isEmpty()) {
                return null;
            }

            IjkMediaCodecInfo bestCodec = candidateCodecList.get(0);

            for (IjkMediaCodecInfo codec : candidateCodecList) {
                if (codec.mRank > bestCodec.mRank) {
                    bestCodec = codec;
                }
            }

            if (bestCodec.mRank < IjkMediaCodecInfo.RANK_LAST_CHANCE) {
                Log.w(TAG, String.format(Locale.US, "unaccetable codec: %s", bestCodec.mCodecInfo.getName()));
                return null;
            }

            Log.i(TAG, String.format(Locale.US, "selected codec: %s rank=%d", bestCodec.mCodecInfo.getName(), bestCodec.mRank));
            return bestCodec.mCodecInfo.getName();
        }
    }

    public static native void native_profileBegin(String libName);

    public static native void native_profileEnd();

    public static native void native_setLogLevel(int level);

    public static native int kantv_anti_remove_rename_this_file();
}
