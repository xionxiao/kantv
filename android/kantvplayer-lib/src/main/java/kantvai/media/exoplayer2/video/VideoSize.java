/*
 * Copyright 2021 The Android Open Source Project
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
package kantvai.media.exoplayer2.video;

import android.os.Bundle;
import androidx.annotation.FloatRange;
import androidx.annotation.IntDef;
import androidx.annotation.IntRange;
import androidx.annotation.Nullable;
import kantvai.media.exoplayer2.Bundleable;
import java.lang.annotation.Documented;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** Represents the video size. */
public final class VideoSize implements Bundleable {

  private static final int DEFAULT_WIDTH = 0;
  private static final int DEFAULT_HEIGHT = 0;
  private static final int DEFAULT_UNAPPLIED_ROTATION_DEGREES = 0;
  private static final float DEFAULT_PIXEL_WIDTH_HEIGHT_RATIO = 1F;

  public static final VideoSize UNKNOWN = new VideoSize(DEFAULT_WIDTH, DEFAULT_HEIGHT);

  /** The video width in pixels, 0 when unknown. */
  @IntRange(from = 0)
  public final int width;

  /** The video height in pixels, 0 when unknown. */
  @IntRange(from = 0)
  public final int height;

  /**
   * Clockwise rotation in degrees that the application should apply for the video for it to be
   * rendered in the correct orientation.
   *
   * <p>Is 0 if unknown or if no rotation is needed.
   *
   * <p>Player should apply video rotation internally, in which case unappliedRotationDegrees is 0.
   * But when a player can't apply the rotation, for example before API level 21, the unapplied
   * rotation is reported by this field for application to handle.
   *
   * <p>Applications that use {@link android.view.TextureView} can apply the rotation by calling
   * {@link android.view.TextureView#setTransform}.
   */
  @IntRange(from = 0, to = 359)
  public final int unappliedRotationDegrees;

  /**
   * The width to height ratio of each pixel, 1 if unknown.
   *
   * <p>For the normal case of square pixels this will be equal to 1.0. Different values are
   * indicative of anamorphic content.
   */
  @FloatRange(from = 0, fromInclusive = false)
  public final float pixelWidthHeightRatio;

  /**
   * Creates a VideoSize without unapplied rotation or anamorphic content.
   *
   * @param width The video width in pixels.
   * @param height The video height in pixels.
   */
  public VideoSize(@IntRange(from = 0) int width, @IntRange(from = 0) int height) {
    this(width, height, DEFAULT_UNAPPLIED_ROTATION_DEGREES, DEFAULT_PIXEL_WIDTH_HEIGHT_RATIO);
  }

  /**
   * Creates a VideoSize.
   *
   * @param width The video width in pixels.
   * @param height The video height in pixels.
   * @param unappliedRotationDegrees Clockwise rotation in degrees that the application should apply
   *     for the video for it to be rendered in the correct orientation. See {@link
   *     #unappliedRotationDegrees}.
   * @param pixelWidthHeightRatio The width to height ratio of each pixel. For the normal case of
   *     square pixels this will be equal to 1.0. Different values are indicative of anamorphic
   *     content.
   */
  public VideoSize(
      @IntRange(from = 0) int width,
      @IntRange(from = 0) int height,
      @IntRange(from = 0, to = 359) int unappliedRotationDegrees,
      @FloatRange(from = 0, fromInclusive = false) float pixelWidthHeightRatio) {
    this.width = width;
    this.height = height;
    this.unappliedRotationDegrees = unappliedRotationDegrees;
    this.pixelWidthHeightRatio = pixelWidthHeightRatio;
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj instanceof VideoSize) {
      VideoSize other = (VideoSize) obj;
      return width == other.width
          && height == other.height
          && unappliedRotationDegrees == other.unappliedRotationDegrees
          && pixelWidthHeightRatio == other.pixelWidthHeightRatio;
    }
    return false;
  }

  @Override
  public int hashCode() {
    int result = 7;
    result = 31 * result + width;
    result = 31 * result + height;
    result = 31 * result + unappliedRotationDegrees;
    result = 31 * result + Float.floatToRawIntBits(pixelWidthHeightRatio);
    return result;
  }

  // Bundleable implementation.
  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({
    FIELD_WIDTH,
    FIELD_HEIGHT,
    FIELD_UNAPPLIED_ROTATION_DEGREES,
    FIELD_PIXEL_WIDTH_HEIGHT_RATIO,
  })
  private @interface FieldNumber {}

  private static final int FIELD_WIDTH = 0;
  private static final int FIELD_HEIGHT = 1;
  private static final int FIELD_UNAPPLIED_ROTATION_DEGREES = 2;
  private static final int FIELD_PIXEL_WIDTH_HEIGHT_RATIO = 3;

  @Override
  public Bundle toBundle() {
    Bundle bundle = new Bundle();
    bundle.putInt(keyForField(FIELD_WIDTH), width);
    bundle.putInt(keyForField(FIELD_HEIGHT), height);
    bundle.putInt(keyForField(FIELD_UNAPPLIED_ROTATION_DEGREES), unappliedRotationDegrees);
    bundle.putFloat(keyForField(FIELD_PIXEL_WIDTH_HEIGHT_RATIO), pixelWidthHeightRatio);
    return bundle;
  }

  public static final Creator<VideoSize> CREATOR =
      bundle -> {
        int width = bundle.getInt(keyForField(FIELD_WIDTH), DEFAULT_WIDTH);
        int height = bundle.getInt(keyForField(FIELD_HEIGHT), DEFAULT_HEIGHT);
        int unappliedRotationDegrees =
            bundle.getInt(
                keyForField(FIELD_UNAPPLIED_ROTATION_DEGREES), DEFAULT_UNAPPLIED_ROTATION_DEGREES);
        float pixelWidthHeightRatio =
            bundle.getFloat(
                keyForField(FIELD_PIXEL_WIDTH_HEIGHT_RATIO), DEFAULT_PIXEL_WIDTH_HEIGHT_RATIO);
        return new VideoSize(width, height, unappliedRotationDegrees, pixelWidthHeightRatio);
      };

  private static String keyForField(@FieldNumber int field) {
    return Integer.toString(field, Character.MAX_RADIX);
  }
}
