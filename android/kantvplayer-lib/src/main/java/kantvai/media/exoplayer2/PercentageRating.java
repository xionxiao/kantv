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
package kantvai.media.exoplayer2;

import static kantvai.media.exoplayer2.util.Assertions.checkArgument;

import android.os.Bundle;
import androidx.annotation.FloatRange;
import androidx.annotation.IntDef;
import androidx.annotation.Nullable;
import com.google.common.base.Objects;
import java.lang.annotation.Documented;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** A rating expressed as a percentage. */
public final class PercentageRating extends Rating {

  private final float percent;

  /** Creates a unrated instance. */
  public PercentageRating() {
    percent = RATING_UNSET;
  }

  /**
   * Creates a rated instance with the given percentage.
   *
   * @param percent The percentage value of the rating.
   */
  public PercentageRating(@FloatRange(from = 0, to = 100) float percent) {
    checkArgument(percent >= 0.0f && percent <= 100.0f, "percent must be in the range of [0, 100]");
    this.percent = percent;
  }

  @Override
  public boolean isRated() {
    return percent != RATING_UNSET;
  }

  /**
   * Returns the percent value of this rating. Will be within the range {@code [0f, 100f]}, or
   * {@link #RATING_UNSET} if unrated.
   */
  public float getPercent() {
    return percent;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(percent);
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (!(obj instanceof PercentageRating)) {
      return false;
    }
    return percent == ((PercentageRating) obj).percent;
  }

  // Bundleable implementation.

  @RatingType private static final int TYPE = RATING_TYPE_PERCENTAGE;

  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({FIELD_RATING_TYPE, FIELD_PERCENT})
  private @interface FieldNumber {}

  private static final int FIELD_PERCENT = 1;

  @Override
  public Bundle toBundle() {
    Bundle bundle = new Bundle();
    bundle.putInt(keyForField(FIELD_RATING_TYPE), TYPE);
    bundle.putFloat(keyForField(FIELD_PERCENT), percent);
    return bundle;
  }

  /** Object that can restore a {@link PercentageRating} from a {@link Bundle}. */
  public static final Creator<PercentageRating> CREATOR = PercentageRating::fromBundle;

  private static PercentageRating fromBundle(Bundle bundle) {
    checkArgument(
        bundle.getInt(keyForField(FIELD_RATING_TYPE), /* defaultValue= */ RATING_TYPE_DEFAULT)
            == TYPE);
    float percent = bundle.getFloat(keyForField(FIELD_PERCENT), /* defaultValue= */ RATING_UNSET);
    return percent == RATING_UNSET ? new PercentageRating() : new PercentageRating(percent);
  }

  private static String keyForField(@FieldNumber int field) {
    return Integer.toString(field, Character.MAX_RADIX);
  }
}
