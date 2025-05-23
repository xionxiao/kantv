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
package kantvai.media.exoplayer2.source.hls;

import androidx.annotation.VisibleForTesting;
import kantvai.media.exoplayer2.Format;
import kantvai.media.exoplayer2.extractor.Extractor;
import kantvai.media.exoplayer2.extractor.ExtractorInput;
import kantvai.media.exoplayer2.extractor.ExtractorOutput;
import kantvai.media.exoplayer2.extractor.PositionHolder;
import kantvai.media.exoplayer2.extractor.mp3.Mp3Extractor;
import kantvai.media.exoplayer2.extractor.mp4.FragmentedMp4Extractor;
import kantvai.media.exoplayer2.extractor.ts.Ac3Extractor;
import kantvai.media.exoplayer2.extractor.ts.Ac4Extractor;
import kantvai.media.exoplayer2.extractor.ts.AdtsExtractor;
import kantvai.media.exoplayer2.extractor.ts.TsExtractor;
import kantvai.media.exoplayer2.util.Assertions;
import kantvai.media.exoplayer2.util.TimestampAdjuster;
import java.io.IOException;

/**
 * {@link HlsMediaChunkExtractor} implementation that uses ExoPlayer app-bundled {@link Extractor
 * Extractors}.
 */
public final class BundledHlsMediaChunkExtractor implements HlsMediaChunkExtractor {

  private static final PositionHolder POSITION_HOLDER = new PositionHolder();

  @VisibleForTesting /* package */ final Extractor extractor;
  private final Format masterPlaylistFormat;
  private final TimestampAdjuster timestampAdjuster;

  /**
   * Creates a new instance.
   *
   * @param extractor The underlying {@link Extractor}.
   * @param masterPlaylistFormat The {@link Format} obtained from the master playlist.
   * @param timestampAdjuster A {@link TimestampAdjuster} to adjust sample timestamps.
   */
  public BundledHlsMediaChunkExtractor(
      Extractor extractor, Format masterPlaylistFormat, TimestampAdjuster timestampAdjuster) {
    this.extractor = extractor;
    this.masterPlaylistFormat = masterPlaylistFormat;
    this.timestampAdjuster = timestampAdjuster;
  }

  @Override
  public void init(ExtractorOutput extractorOutput) {
    extractor.init(extractorOutput);
  }

  @Override
  public boolean read(ExtractorInput extractorInput) throws IOException {
    return extractor.read(extractorInput, POSITION_HOLDER) == Extractor.RESULT_CONTINUE;
  }

  @Override
  public boolean isPackedAudioExtractor() {
    return extractor instanceof AdtsExtractor
        || extractor instanceof Ac3Extractor
        || extractor instanceof Ac4Extractor
        || extractor instanceof Mp3Extractor;
  }

  @Override
  public boolean isReusable() {
    return extractor instanceof TsExtractor || extractor instanceof FragmentedMp4Extractor;
  }

  @Override
  public HlsMediaChunkExtractor recreate() {
    Assertions.checkState(!isReusable());
    Extractor newExtractorInstance;
    if (extractor instanceof WebvttExtractor) {
      newExtractorInstance = new WebvttExtractor(masterPlaylistFormat.language, timestampAdjuster);
    } else if (extractor instanceof AdtsExtractor) {
      newExtractorInstance = new AdtsExtractor();
    } else if (extractor instanceof Ac3Extractor) {
      newExtractorInstance = new Ac3Extractor();
    } else if (extractor instanceof Ac4Extractor) {
      newExtractorInstance = new Ac4Extractor();
    } else if (extractor instanceof Mp3Extractor) {
      newExtractorInstance = new Mp3Extractor();
    } else {
      throw new IllegalStateException(
          "Unexpected extractor type for recreation: " + extractor.getClass().getSimpleName());
    }
    return new BundledHlsMediaChunkExtractor(
        newExtractorInstance, masterPlaylistFormat, timestampAdjuster);
  }

  @Override
  public void onTruncatedSegmentParsed() {
    extractor.seek(/* position= */ 0, /* timeUs= */ 0);
  }
}
