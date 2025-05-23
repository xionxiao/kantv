/*
 * Copyright (C) 2015 Bilibili
 * Copyright (C) 2015 Zhang Rui <bbcallen@gmail.com>
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

import android.annotation.TargetApi;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.view.Surface;
import android.view.SurfaceHolder;

import java.io.FileDescriptor;
import java.io.IOException;
import java.text.DateFormat;
import java.util.Map;
import java.util.TimeZone;

import kantvai.media.player.bean.TrackInfoBean;
import kantvai.media.player.misc.IMediaDataSource;
import kantvai.media.player.misc.ITrackInfo;

public class MediaPlayerProxy implements IMediaPlayer {
    protected final IMediaPlayer mBackEndMediaPlayer;

    public MediaPlayerProxy(IMediaPlayer backEndMediaPlayer) {
        mBackEndMediaPlayer = backEndMediaPlayer;
    }

    public IMediaPlayer getInternalMediaPlayer() {
        return mBackEndMediaPlayer;
    }

    @Override
    public void setDisplay(SurfaceHolder sh) {
        mBackEndMediaPlayer.setDisplay(sh);
    }

    @TargetApi(Build.VERSION_CODES.ICE_CREAM_SANDWICH)
    @Override
    public void setSurface(Surface surface) {
        mBackEndMediaPlayer.setSurface(surface);
    }

    @Override
    public void setDataSource(Context context, Uri uri)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException {
        mBackEndMediaPlayer.setDataSource(context, uri);
    }

    @TargetApi(Build.VERSION_CODES.ICE_CREAM_SANDWICH)
    @Override
    public void setDataSource(Context context, Uri uri, Map<String, String> headers)
            throws IOException, IllegalArgumentException, SecurityException, IllegalStateException {
        mBackEndMediaPlayer.setDataSource(context, uri, headers);
    }

    @Override
    public void setDataSource(FileDescriptor fd)
            throws IOException, IllegalArgumentException, IllegalStateException {
        mBackEndMediaPlayer.setDataSource(fd);
    }

    @Override
    public void setDataSource(String path) throws IOException, IllegalArgumentException, SecurityException, IllegalStateException {
        mBackEndMediaPlayer.setDataSource(path);
    }

    @Override
    public void setDataSource(IMediaDataSource mediaDataSource)  {
        mBackEndMediaPlayer.setDataSource(mediaDataSource);
    }

    @Override
    public void setVideoTitle(String title) {
        mBackEndMediaPlayer.setVideoTitle(title);
    }

    @Override
    public String getVideoTitle() {
        return  mBackEndMediaPlayer.getVideoTitle();
    }

    @Override
    public String getDataSource() {
        return mBackEndMediaPlayer.getDataSource();
    }

    @Override
    public void prepareAsync() throws IllegalStateException {
        mBackEndMediaPlayer.prepareAsync();
    }

    @Override
    public void start() throws IllegalStateException {
        mBackEndMediaPlayer.start();
    }

    @Override
    public void stop() throws IllegalStateException {
        mBackEndMediaPlayer.stop();
    }

    @Override
    public void pause() throws IllegalStateException {
        mBackEndMediaPlayer.pause();
    }

    @Override
    public void setScreenOnWhilePlaying(boolean screenOn) {
        mBackEndMediaPlayer.setScreenOnWhilePlaying(screenOn);
    }

    @Override
    public int getVideoWidth() {
        return mBackEndMediaPlayer.getVideoWidth();
    }

    @Override
    public int getVideoHeight() {
        return mBackEndMediaPlayer.getVideoHeight();
    }

    @Override
    public boolean isPlaying() {
        return mBackEndMediaPlayer.isPlaying();
    }

    @Override
    public boolean isClearContent() {
        return mBackEndMediaPlayer.isClearContent();
    }

    @Override
    public long getBitRate() {
        return mBackEndMediaPlayer.getBitRate();
    }

    @Override
    public void setDisableAudioTrack(boolean bDisable) {
        mBackEndMediaPlayer.setDisableAudioTrack(bDisable);
    }


    @Override
    public void setDisableVideoTrack(boolean bDisable) {
        mBackEndMediaPlayer.setDisableVideoTrack(bDisable);
    }

    @Override
    public void setEnableDumpVideoES(boolean bEnable) {
        mBackEndMediaPlayer.setEnableDumpVideoES(bEnable);
    }


    @Override
    public void setEnableDumpAudioES(boolean bEnable) {
        mBackEndMediaPlayer.setEnableDumpAudioES(bEnable);
    }


    @Override
    public void setEnableRecord(boolean bEnableRecord) {
        mBackEndMediaPlayer.setEnableRecord(bEnableRecord);
    }


    @Override
    public void setEnableASR(boolean bEnableASR) {
        mBackEndMediaPlayer.setEnableASR(bEnableASR);
    }

    @Override
    public void updateRecordingStatus(boolean bEnableRecord) {
        mBackEndMediaPlayer.updateRecordingStatus(bEnableRecord);
    }


    @Override
    public boolean getEnableRecord() {
        return false;
    }


    @Override
    public boolean getEnableASR() {
        return false;
    }


    @Override
    public void setDurationDumpES(int durationDumpES) {
        mBackEndMediaPlayer.setDurationDumpES(durationDumpES);
    }

    @Override
    public void setDecryptMode(int decryptMode) {
        mBackEndMediaPlayer.setDecryptMode(decryptMode);
    }

    @Override
    public String getDrmInfo()  { return mBackEndMediaPlayer.getDrmInfo(); }

    @Override
    public void selectTrack(TrackInfoBean trackInfo, boolean isAudio) {
        mBackEndMediaPlayer.selectTrack(trackInfo, isAudio);
    }

    @Override
    public String getVideoInfo() {
        return mBackEndMediaPlayer.getVideoInfo();
    }

    @Override
    public String getAudioInfo() {
        return mBackEndMediaPlayer.getAudioInfo();
    }

    @Override
    public String getVideoDecoderName() {
        return mBackEndMediaPlayer.getVideoDecoderName();
    }

    @Override
    public String getAudioDecoderName() {
        return mBackEndMediaPlayer.getAudioDecoderName();
    }


    @Override
    public long getBeginTime() {
        return mBackEndMediaPlayer.getBeginTime();
    }

    @Override
    public long getPauseTime() {
        return mBackEndMediaPlayer.getPauseTime();
    }

    @Override
    public long getPlayTime() {
        return mBackEndMediaPlayer.getPlayTime();
    }
    //end added

    @Override
    public void seekTo(long msec) throws IllegalStateException {
        mBackEndMediaPlayer.seekTo(msec);
    }

    @Override
    public long getCurrentPosition() {
        return mBackEndMediaPlayer.getCurrentPosition();
    }

    @Override
    public long getDuration() {
        return mBackEndMediaPlayer.getDuration();
    }

    @Override
    public void release() {
        mBackEndMediaPlayer.release();
    }

    @Override
    public void reset() {
        mBackEndMediaPlayer.reset();
    }

    @Override
    public void setVolume(float leftVolume, float rightVolume) {
        mBackEndMediaPlayer.setVolume(leftVolume, rightVolume);
    }

    @Override
    public int getAudioSessionId() {
        return mBackEndMediaPlayer.getAudioSessionId();
    }

    @Override
    public IjkMediaInfo getMediaInfo() {
        return mBackEndMediaPlayer.getMediaInfo();
    }

    @Override
    public void setLogEnabled(boolean enable) {

    }

    @Override
    public boolean isPlayable() {
        return false;
    }

    @Override
    public void setOnPreparedListener(OnPreparedListener listener) {
        if (listener != null) {
            final OnPreparedListener finalListener = listener;
            mBackEndMediaPlayer.setOnPreparedListener(new OnPreparedListener() {
                @Override
                public void onPrepared(IMediaPlayer mp) {
                    finalListener.onPrepared(MediaPlayerProxy.this);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnPreparedListener(null);
        }
    }

    @Override
    public void setOnCompletionListener(OnCompletionListener listener) {
        if (listener != null) {
            final OnCompletionListener finalListener = listener;
            mBackEndMediaPlayer.setOnCompletionListener(new OnCompletionListener() {
                @Override
                public void onCompletion(IMediaPlayer mp) {
                    finalListener.onCompletion(MediaPlayerProxy.this);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnCompletionListener(null);
        }
    }

    @Override
    public void setOnBufferingUpdateListener(OnBufferingUpdateListener listener) {
        if (listener != null) {
            final OnBufferingUpdateListener finalListener = listener;
            mBackEndMediaPlayer.setOnBufferingUpdateListener(new OnBufferingUpdateListener() {
                @Override
                public void onBufferingUpdate(IMediaPlayer mp, int percent) {
                    finalListener.onBufferingUpdate(MediaPlayerProxy.this, percent);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnBufferingUpdateListener(null);
        }
    }

    @Override
    public void setOnSeekCompleteListener(OnSeekCompleteListener listener) {
        if (listener != null) {
            final OnSeekCompleteListener finalListener = listener;
            mBackEndMediaPlayer.setOnSeekCompleteListener(new OnSeekCompleteListener() {
                @Override
                public void onSeekComplete(IMediaPlayer mp) {
                    finalListener.onSeekComplete(MediaPlayerProxy.this);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnSeekCompleteListener(null);
        }
    }

    @Override
    public void setOnVideoSizeChangedListener(OnVideoSizeChangedListener listener) {
        if (listener != null) {
            final OnVideoSizeChangedListener finalListener = listener;
            mBackEndMediaPlayer.setOnVideoSizeChangedListener(new OnVideoSizeChangedListener() {
                @Override
                public void onVideoSizeChanged(IMediaPlayer mp, int width, int height, int sar_num, int sar_den) {
                    finalListener.onVideoSizeChanged(MediaPlayerProxy.this, width, height, sar_num, sar_den);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnVideoSizeChangedListener(null);
        }
    }

    @Override
    public void setOnErrorListener(OnErrorListener listener) {
        if (listener != null) {
            final OnErrorListener finalListener = listener;
            mBackEndMediaPlayer.setOnErrorListener(new OnErrorListener() {
                @Override
                public boolean onError(IMediaPlayer mp, int what, int extra) {
                    return finalListener.onError(MediaPlayerProxy.this, what, extra);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnErrorListener(null);
        }
    }

    @Override
    public void setOnInfoListener(OnInfoListener listener) {
        if (listener != null) {
            final OnInfoListener finalListener = listener;
            mBackEndMediaPlayer.setOnInfoListener(new OnInfoListener() {
                @Override
                public boolean onInfo(IMediaPlayer mp, int what, int extra) {
                    return finalListener.onInfo(MediaPlayerProxy.this, what, extra);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnInfoListener(null);
        }
    }

    @Override
    public void setOnTimedTextListener(OnTimedTextListener listener) {
        if (listener != null) {
            final OnTimedTextListener finalListener = listener;
            mBackEndMediaPlayer.setOnTimedTextListener(new OnTimedTextListener() {
                @Override
                public void onTimedText(IMediaPlayer mp, IjkTimedText text) {
                    finalListener.onTimedText(MediaPlayerProxy.this, text);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnTimedTextListener(null);
        }
    }

    @Override
    public  void setOnTrackChangeListener(OnTrackChangeListener listener) {
        if (listener != null) {
            final OnTrackChangeListener finalListener = listener;
            mBackEndMediaPlayer.setOnTrackChangeListener(new OnTrackChangeListener() {
                @Override
                public void onTrackChange(IMediaPlayer mp, Object trackSelector, Object trackGroups, Object trackSelections) {
                    finalListener.onTrackChange(MediaPlayerProxy.this, trackSelector, trackGroups, trackSelections);
                }
            });
        } else {
            mBackEndMediaPlayer.setOnTrackChangeListener(null);
        }
    }

    @Override
    public void setAudioStreamType(int streamtype) {
        mBackEndMediaPlayer.setAudioStreamType(streamtype);
    }

    @Override
    public void setKeepInBackground(boolean keepInBackground) {
        mBackEndMediaPlayer.setKeepInBackground(keepInBackground);
    }

    @Override
    public int getVideoSarNum() {
        return mBackEndMediaPlayer.getVideoSarNum();
    }

    @Override
    public int getVideoSarDen() {
        return mBackEndMediaPlayer.getVideoSarDen();
    }

    @Override
    public void setWakeMode(Context context, int mode) {
        mBackEndMediaPlayer.setWakeMode(context, mode);
    }

    @Override
    public ITrackInfo[] getTrackInfo() {
        return mBackEndMediaPlayer.getTrackInfo();
    }

    @Override
    public void setLooping(boolean looping) {
        mBackEndMediaPlayer.setLooping(looping);
    }

    @Override
    public boolean isLooping() {
        return mBackEndMediaPlayer.isLooping();
    }
}
