/*
 * Copyright (C) 2006 The Android Open Source Project
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
package kantvai.media.exoplayer2.kantvexo2;

import android.content.Context;
import android.os.Build;

import kantvai.media.exoplayer2.DefaultRenderersFactory;
import kantvai.media.exoplayer2.ExoPlayerLibraryInfo;
import kantvai.media.exoplayer2.RenderersFactory;
import kantvai.media.exoplayer2.database.DatabaseProvider;
import kantvai.media.exoplayer2.database.ExoDatabaseProvider;
import kantvai.media.exoplayer2.offline.ActionFileUpgradeUtil;
import kantvai.media.exoplayer2.offline.DefaultDownloadIndex;
import kantvai.media.exoplayer2.offline.DownloadManager;
import kantvai.media.exoplayer2.upstream.DataSource;
import kantvai.media.exoplayer2.upstream.DefaultDataSourceFactory;
import kantvai.media.exoplayer2.upstream.DefaultHttpDataSource;
import kantvai.media.exoplayer2.upstream.HttpDataSource;
import kantvai.media.exoplayer2.upstream.cache.Cache;
import kantvai.media.exoplayer2.upstream.cache.CacheDataSource;
import kantvai.media.exoplayer2.upstream.cache.NoOpCacheEvictor;
import kantvai.media.exoplayer2.upstream.cache.SimpleCache;
import kantvai.media.exoplayer2.util.Log;

import java.io.File;
import java.io.IOException;
import java.net.CookieHandler;
import java.net.CookieManager;
import java.net.CookiePolicy;
import java.util.concurrent.Executors;

import kantvai.media.exoplayer2.ext.cronet.CronetDataSource;
import kantvai.media.exoplayer2.ext.cronet.CronetUtil;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
/*
 * Copyright 2020 The Android Open Source Project
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
import org.checkerframework.checker.nullness.qual.Nullable;
import org.chromium.net.CronetEngine;

import kantvai.media.player.KANTVLog;
import kantvai.media.player.KANTVUtils;


/** Utility methods for the demo app. */
public final class DemoUtil {

    public static final String DOWNLOAD_NOTIFICATION_CHANNEL_ID = "download_channel";

    /**
     * Whether the demo application uses Cronet for networking. Note that Cronet does not provide
     * automatic support for cookies (https://github.com/google/ExoPlayer/issues/5975).
     *
     * <p>If set to false, the platform's default network stack is used with a {@link CookieManager}
     * configured in {@link #getHttpDataSourceFactory}.
     */
    private static final boolean USE_CRONET_FOR_NETWORKING = true;

    private static final String USER_AGENT =
            "ExoPlayerDemo/"
                    + ExoPlayerLibraryInfo.VERSION
                    + " (Linux; Android "
                    + Build.VERSION.RELEASE
                    + ") "
                    + ExoPlayerLibraryInfo.VERSION_SLASHY;
    private static final String TAG = "DemoUtil";
    private static final String DOWNLOAD_ACTION_FILE = "actions";
    private static final String DOWNLOAD_TRACKER_ACTION_FILE = "tracked_actions";
    private static final String DOWNLOAD_CONTENT_DIRECTORY = "downloads";

    private static DataSource.@MonotonicNonNull Factory dataSourceFactory;
    private static HttpDataSource.@MonotonicNonNull Factory httpDataSourceFactory;
    private static @MonotonicNonNull
    DatabaseProvider databaseProvider;
    private static @MonotonicNonNull
    File downloadDirectory;
    private static @MonotonicNonNull
    Cache downloadCache;
    private static @MonotonicNonNull
    DownloadManager downloadManager;
    //private static @MonotonicNonNull DownloadTracker downloadTracker;
    //private static @MonotonicNonNull DownloadNotificationHelper downloadNotificationHelper;

    /** Returns whether extension renderers should be used. */
    public static boolean useExtensionRenderers() {
        return KANTVUtils.isUseDecoderExtensions();
    }

    public static RenderersFactory buildRenderersFactory(
            Context context, boolean preferExtensionRenderer) {
        @DefaultRenderersFactory.ExtensionRendererMode
        int extensionRendererMode =
                useExtensionRenderers()
                        ? (preferExtensionRenderer
                        ? DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER
                        : DefaultRenderersFactory.EXTENSION_RENDERER_MODE_ON)
                        : DefaultRenderersFactory.EXTENSION_RENDERER_MODE_OFF;
        return new DefaultRenderersFactory(context.getApplicationContext())
                .setExtensionRendererMode(extensionRendererMode);
    }

    public static synchronized HttpDataSource.Factory getHttpDataSourceFactory(Context context) {
        if (httpDataSourceFactory == null) {
            if (USE_CRONET_FOR_NETWORKING) {
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                KANTVLog.d(TAG, "USE_CRONET_FOR_NETWORKING");
                context = context.getApplicationContext();
                @Nullable
                CronetEngine cronetEngine =
                        CronetUtil.buildCronetEngine(context, USER_AGENT, /* preferGMSCoreCronet= */ false);
                if (cronetEngine != null) {
                    httpDataSourceFactory =
                            new CronetDataSource.Factory(cronetEngine, Executors.newSingleThreadExecutor());
                }
            }
            if (httpDataSourceFactory == null) {
                // We don't want to use Cronet, or we failed to instantiate a CronetEngine.
                CookieManager cookieManager = new CookieManager();
                cookieManager.setCookiePolicy(CookiePolicy.ACCEPT_ORIGINAL_SERVER);
                CookieHandler.setDefault(cookieManager);
                httpDataSourceFactory = new DefaultHttpDataSource.Factory().setUserAgent(USER_AGENT);
            }
        }
        return httpDataSourceFactory;
    }

    /** Returns a {@link DataSource.Factory}. */
    public static synchronized DataSource.Factory getDataSourceFactory(Context context) {
        if (dataSourceFactory == null) {
            context = context.getApplicationContext();
            DefaultDataSourceFactory upstreamFactory =
                    new DefaultDataSourceFactory(context, getHttpDataSourceFactory(context));
            dataSourceFactory = buildReadOnlyCacheDataSource(upstreamFactory, getDownloadCache(context));
        }
        return dataSourceFactory;
    }

    /*
    public static synchronized DownloadNotificationHelper getDownloadNotificationHelper(
            Context context) {
        if (downloadNotificationHelper == null) {
            downloadNotificationHelper =
                    new DownloadNotificationHelper(context, DOWNLOAD_NOTIFICATION_CHANNEL_ID);
        }
        return downloadNotificationHelper;
    }


    public static synchronized DownloadTracker getDownloadTracker(Context context) {
        ensureDownloadManagerInitialized(context);
        return downloadTracker;
        return null;
    }
    */

    public static synchronized DownloadManager getDownloadManager(Context context) {
        ensureDownloadManagerInitialized(context);
        return downloadManager;
    }

    private static synchronized Cache getDownloadCache(Context context) {
        if (downloadCache == null) {
            File downloadContentDirectory =
                    new File(getDownloadDirectory(context), DOWNLOAD_CONTENT_DIRECTORY);
            downloadCache =
                    new SimpleCache(
                            downloadContentDirectory, new NoOpCacheEvictor(), getDatabaseProvider(context));
        }
        return downloadCache;
    }

    private static synchronized void ensureDownloadManagerInitialized(Context context) {
        if (downloadManager == null) {
            DefaultDownloadIndex downloadIndex = new DefaultDownloadIndex(getDatabaseProvider(context));
            upgradeActionFile(
                    context, DOWNLOAD_ACTION_FILE, downloadIndex, /* addNewDownloadsAsCompleted= */ false);
            upgradeActionFile(
                    context,
                    DOWNLOAD_TRACKER_ACTION_FILE,
                    downloadIndex,
                    /* addNewDownloadsAsCompleted= */ true);
            downloadManager =
                    new DownloadManager(
                            context,
                            getDatabaseProvider(context),
                            getDownloadCache(context),
                            getHttpDataSourceFactory(context),
                            Executors.newFixedThreadPool(/* nThreads= */ 6));
            //downloadTracker =
            //        new DownloadTracker(context, getHttpDataSourceFactory(context), downloadManager);
        }
    }

    private static synchronized void upgradeActionFile(
            Context context,
            String fileName,
            DefaultDownloadIndex downloadIndex,
            boolean addNewDownloadsAsCompleted) {
        try {
            ActionFileUpgradeUtil.upgradeAndDelete(
                    new File(getDownloadDirectory(context), fileName),
                    /* downloadIdProvider= */ null,
                    downloadIndex,
                    /* deleteOnFailure= */ true,
                    addNewDownloadsAsCompleted);
        } catch (IOException e) {
            Log.e(TAG, "Failed to upgrade action file: " + fileName, e);
        }
    }

    private static synchronized DatabaseProvider getDatabaseProvider(Context context) {
        if (databaseProvider == null) {
            databaseProvider = new ExoDatabaseProvider(context);
        }
        return databaseProvider;
    }

    private static synchronized File getDownloadDirectory(Context context) {
        if (downloadDirectory == null) {
            downloadDirectory = context.getExternalFilesDir(/* type= */ null);
            if (downloadDirectory == null) {
                downloadDirectory = context.getFilesDir();
            }
        }
        return downloadDirectory;
    }

    private static CacheDataSource.Factory buildReadOnlyCacheDataSource(
            DataSource.Factory upstreamFactory, Cache cache) {
        return new CacheDataSource.Factory()
                .setCache(cache)
                .setUpstreamDataSourceFactory(upstreamFactory)
                .setCacheWriteDataSinkFactory(null)
                .setFlags(CacheDataSource.FLAG_IGNORE_CACHE_ON_ERROR);
    }

    private DemoUtil() {}
}
