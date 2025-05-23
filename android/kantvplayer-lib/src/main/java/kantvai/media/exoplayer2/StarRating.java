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
import androidx.annotation.IntRange;
import androidx.annotation.Nullable;
import com.google.common.base.Objects;
import java.lang.annotation.Documented;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** A rating expressed as a fractional number of stars. */
public final class StarRating extends Rating {

  @IntRange(from = 1)
  private final int maxStars;

  private final float starRating;

  /**
   * Creates a unrated instance with {@code maxStars}. If {@code maxStars} is not a positive
   * integer, it will throw an {@link IllegalArgumentException}.
   *
   * @param maxStars The maximum number of stars this rating can have.
   */
  public StarRating(@IntRange(from = 1) int maxStars) {
    checkArgument(maxStars > 0, "maxStars must be a positive integer");
    this.maxStars = maxStars;
    starRating = RATING_UNSET;
  }

  /**
   * Creates a rated instance with {@code maxStars} and the given fractional number of stars.
   * Non-integer values may be used to represent an average rating value. If {@code maxStars} is not
   * a positive integer or {@code starRating} is out of range, it will throw an {@link
   * IllegalArgumentException}.
   *
   * @param maxStars The maximum number of stars this rating can have.
   * @param starRating A fractional number of stars of this rating from {@code 0f} to {@code
   *     maxStars}.
   */
  public StarRating(@IntRange(from = 1) int maxStars, @FloatRange(from = 0.0) float starRating) {
    checkArgument(maxStars > 0, "maxStars must be a positive integer");
    checkArgument(
        starRating >= 0.0f && starRating <= maxStars, "starRating is out of range [0, maxStars]");
    this.maxStars = maxStars;
    this.starRating = starRating;
  }

  @Override
  public boolean isRated() {
    return starRating != RATING_UNSET;
  }

  /** Returns the maximum number of stars. Must be a positive number. */
  @IntRange(from = 1)
  public int getMaxStars() {
    return maxStars;
  }

  /**
   * Returns the fractional number of stars of this rating. Will range from {@code 0f} to {@link
   * #maxStars}, or {@link #RATING_UNSET} if unrated.
   */
  public float getStarRating() {
    return starRating;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(maxStars, starRating);
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (!(obj instanceof StarRating)) {
      return false;
    }
    StarRating other = (StarRating) obj;
    return maxStars == other.maxStars && starRating == other.starRating;
  }

  // Bundleable implementation.

  @RatingType private static final int TYPE = RATING_TYPE_STAR;
  private static final int MAX_STARS_DEFAULT = 5;

  @Documented
  @Retention(RetentionPolicy.SOURCE)
  @IntDef({FIELD_RATING_TYPE, FIELD_MAX_STARS, FIELD_STAR_RATING})
  private @interface FieldNumber {}

  private static final int FIELD_MAX_STARS = 1;
  private static final int FIELD_STAR_RATING = 2;

  @Override
  public Bundle toBundle() {
    Bundle bundle = new Bundle();
    bundle.putInt(keyForField(FIELD_RATING_TYPE), TYPE);
    bundle.putInt(keyForField(FIELD_MAX_STARS), maxStars);
    bundle.putFloat(keyForField(FIELD_STAR_RATING), starRating);
    return bundle;
  }

  /** Object that can restore a {@link StarRating} from a {@link Bundle}. */
  public static final Creator<StarRating> CREATOR = StarRating::fromBundle;

  private static StarRating fromBundle(Bundle bundle) {
    checkArgument(
        bundle.getInt(keyForField(FIELD_RATING_TYPE), /* defaultValue= */ RATING_TYPE_DEFAULT)
            == TYPE);
    int maxStars =
        bundle.getInt(keyForField(FIELD_MAX_STARS), /* defaultValue= */ MAX_STARS_DEFAULT);
    float starRating =
        bundle.getFloat(keyForField(FIELD_STAR_RATING), /* defaultValue= */ RATING_UNSET);
    return starRating == RATING_UNSET
        ? new StarRating(maxStars)
        : new StarRating(maxStars, starRating);
  }

  private static String keyForField(@FieldNumber int field) {
    return Integer.toString(field, Character.MAX_RADIX);
  }
}
