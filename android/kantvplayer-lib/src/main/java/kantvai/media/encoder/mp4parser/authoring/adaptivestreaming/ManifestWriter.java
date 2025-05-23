/*
 * Copyright 2012 Sebastian Annies, Hamburg
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package kantvai.media.encoder.mp4parser.authoring.adaptivestreaming;


import kantvai.media.encoder.mp4parser.authoring.Movie;
import kantvai.media.encoder.mp4parser.authoring.Track;

import java.io.IOException;

public interface ManifestWriter {
    String getManifest(Movie inputs) throws IOException;

    long getBitrate(Track track);

    long[] calculateFragmentDurations(Track track, Movie movie);

}
