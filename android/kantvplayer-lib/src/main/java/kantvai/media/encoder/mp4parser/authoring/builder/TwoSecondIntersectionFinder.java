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
package kantvai.media.encoder.mp4parser.authoring.builder;

import kantvai.media.encoder.iso.boxes.TimeToSampleBox;
import kantvai.media.encoder.mp4parser.authoring.Movie;
import kantvai.media.encoder.mp4parser.authoring.Track;

import java.util.Arrays;
import java.util.List;

/**
 * This <code>FragmentIntersectionFinder</code> cuts the input movie in 2 second
 * snippets.
 */
public class TwoSecondIntersectionFinder implements FragmentIntersectionFinder {

    protected long getDuration(Track track) {
        long duration = 0;
        for (TimeToSampleBox.Entry entry : track.getDecodingTimeEntries()) {
            duration += entry.getCount() * entry.getDelta();
        }
        return duration;
    }

    /**
     * {@inheritDoc}
     */
    public long[] sampleNumbers(Track track, Movie movie) {
        List<TimeToSampleBox.Entry> entries = track.getDecodingTimeEntries();

        double trackLength = 0;
        for (Track thisTrack : movie.getTracks()) {
            double thisTracksLength = getDuration(thisTrack) / thisTrack.getTrackMetaData().getTimescale();
            if (trackLength < thisTracksLength) {
                trackLength = thisTracksLength;
            }
        }

        int fragmentCount = (int)Math.ceil(trackLength / 2) - 1;
        if (fragmentCount < 1) {
            fragmentCount = 1;
        }

        long fragments[] = new long[fragmentCount];
        Arrays.fill(fragments, -1);
        fragments[0] = 1;

        long time = 0;
        int samples = 0;
        for (TimeToSampleBox.Entry entry : entries) {
            for (int i = 0; i < entry.getCount(); i++) {
                int currentFragment = (int) (time / track.getTrackMetaData().getTimescale() / 2) + 1;
                if (currentFragment >= fragments.length) {
                    break;
                }
                fragments[currentFragment] = samples++ + 1;
                time += entry.getDelta();
            }
        }
        long last = samples + 1;
        // fill all -1 ones.
        for (int i = fragments.length - 1; i >= 0; i--) {
            if (fragments[i] == -1) {
                fragments[i] = last ;
            }
            last = fragments[i];
        }
        return fragments;

    }

}
