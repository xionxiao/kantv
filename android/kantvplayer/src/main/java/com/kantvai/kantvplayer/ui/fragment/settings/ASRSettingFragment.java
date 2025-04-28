 /*
  * Copyright (c) Project KanTV. 2021-2023
  * Copyright (c) 2024- KanTV Authors
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
 package com.kantvai.kantvplayer.ui.fragment.settings;

 import android.content.Context;
 import android.content.Intent;
 import android.content.SharedPreferences;
 import android.os.Bundle;
 import android.preference.PreferenceManager;
 import android.view.View;
 import android.view.WindowManager;
 import android.widget.Toast;

 import androidx.annotation.NonNull;
 import androidx.annotation.Nullable;
 import androidx.preference.CheckBoxPreference;
 import androidx.preference.Preference;


 import com.kantvai.kantvplayer.ui.activities.ShellActivity;
 import com.kantvai.kantvplayer.utils.Settings;
 import com.kantvai.kantvplayer.R;
 import com.kantvai.kantvplayer.ui.activities.WebViewActivity;

 import java.io.File;

 import kantvai.media.player.KANTVLog;
 import kantvai.media.player.KANTVUtils;


 public class ASRSettingFragment extends BaseSettingsFragment {
     private static final String TAG = ASRSettingFragment.class.getName();
     private static ShellActivity mActivity;
     private Context mContext;
     private Context mAppContext;
     private SharedPreferences mSharedPreferences;
     private Settings mSettings;


     @Override
     public String getTitle() {
         return mActivity.getBaseContext().getString(R.string.asr_settings);
     }

     @Override
     public void onAttach(Context context) {
         super.onAttach(context);
     }


     @Override
     public void onCreatePreferences(Bundle bundle, String s) {
         mActivity = ShellActivity.getInstance();
         mAppContext = mActivity.getApplicationContext();
         mSettings = new Settings(mAppContext);
         mContext = mActivity.getBaseContext();
         mSettings.updateUILang(mActivity);

         addPreferencesFromResource(R.xml.settings_asr);
         mSharedPreferences = PreferenceManager.getDefaultSharedPreferences(mAppContext);

         KANTVLog.j(TAG, "dev mode:" + mSettings.getDevMode());
         //NPU or dedicated hardware AI engine in Android device not supported currently
         if (!mSettings.getDevMode()) {
             if (findPreference("pref.voiceapi") != null) {
                 findPreference("pref.voiceapi").setEnabled(true);
             }
         }


     }

     @Override
     public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
         super.onViewCreated(view, savedInstanceState);

         if (findPreference("pref.voiceapi") != null) {
             findPreference("pref.voiceapi").setOnPreferenceClickListener(preference -> {
                 KANTVUtils.showMsgBox(mActivity, "This feature not implemented currently");
                 return false;
             });
         }
     }


     @Override
     public void onDisplayPreferenceDialog(Preference preference) {
         super.onDisplayPreferenceDialog(preference);
     }

     @Override
     public void onResume() {
         mSharedPreferences.registerOnSharedPreferenceChangeListener(mSharedPreferenceChangeListener);
         super.onResume();
     }

     @Override
     public void onPause() {
         mSharedPreferences.unregisterOnSharedPreferenceChangeListener(mSharedPreferenceChangeListener);
         super.onPause();
     }


     private SharedPreferences.OnSharedPreferenceChangeListener mSharedPreferenceChangeListener = new SharedPreferences.OnSharedPreferenceChangeListener() {
         @Override
         public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
             KANTVLog.j(TAG, "key : " + key);
             if (
                 (key.contains("pref.asrmode"))
                 || (key.contains("pref.asrthreadcounts"))
                 || (key.contains("pref.ggmlmodel"))
             ) {
                 KANTVLog.j(TAG, "asrmode: " + mSettings.getASRMode());
                 KANTVLog.j(TAG, "asrthreadCounts " + mSettings.getASRThreadCounts());
                 KANTVLog.j(TAG, "GGML mode: " + mSettings.getGGMLMode());
                 KANTVLog.j(TAG, "GGML mode name: " + KANTVUtils.getGGMLModeString(mSettings.getGGMLMode()));
                 String modelPath = KANTVUtils.getDataPath() + "ggml-" + KANTVUtils.getGGMLModeString(mSettings.getGGMLMode()) + ".bin";
                 KANTVLog.j(TAG, "modelPath:" + modelPath);
                 KANTVUtils.setASRConfig("whispercpp", modelPath, mSettings.getASRThreadCounts(), mSettings.getASRMode());
             }
         }
     };


     @Override
     public boolean onPreferenceTreeClick(Preference preference) {
         String key = preference.getKey();
         KANTVLog.j(TAG, "key : " + key);
         if (preference instanceof CheckBoxPreference) {
             KANTVLog.d(TAG, "preference : " + preference.getKey() + ", status:" + mSharedPreferences.getBoolean(key, false));
         }


         //focus on GGML's model
         if (key.contains("pref.downloadGGMLmodel")) {
             KANTVLog.j(TAG, "download GGML model");
             WindowManager.LayoutParams attributes = mActivity.getWindow().getAttributes();
             attributes.screenBrightness = 1.0f;
             mActivity.getWindow().setAttributes(attributes);
             mActivity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

             String asrAudioFileName = KANTVUtils.getDataPath() + "jfk.wav";
             //String asrModelFileName = KANTVUtils.getDataPath(mContext) + "ggml-tiny.en-q8_0.bin";
             String userChooseModelName = KANTVUtils.getGGMLModeString(mSettings.getGGMLMode());
             KANTVLog.j(TAG, "ggml model name of user choose:" + userChooseModelName);
             String userChooseModelFileName = "ggml-" + userChooseModelName + ".bin";
             File asrAudioFile = new File(asrAudioFileName);
             File asrModelFile = new File(KANTVUtils.getDataPath() + userChooseModelFileName);
             KANTVLog.j(TAG, "asrAudioFile:" + asrAudioFile.getAbsolutePath());
             KANTVLog.j(TAG, "asrModeFile:" + asrModelFile.getAbsolutePath());
             if (
                     (asrAudioFile == null) || (!asrAudioFile.exists())
                             || (asrModelFile == null) || (!asrModelFile.exists())
             ) {
                 DownloadModel manager = new DownloadModel(mActivity);
                 manager.setTitle("begin download GGML model");
                 manager.setModeName("ASR");
                 manager.setModeName("GGML", "jfk.wav", userChooseModelFileName);
                 manager.showUpdateDialog();
             } else {
                 KANTVLog.j(TAG, "GGML model file already exist: " + KANTVUtils.getDataPath() + userChooseModelFileName);
                 Toast.makeText(mContext, "GGML model file already exist: " + KANTVUtils.getDataPath() + userChooseModelFileName, Toast.LENGTH_SHORT).show();
                 KANTVUtils.showMsgBox(mActivity, "GGML model file already exist: " + KANTVUtils.getDataPath() + userChooseModelFileName);
             }


         }

         return true;
     }


     public static native int kantv_anti_remove_rename_this_file();
 }
