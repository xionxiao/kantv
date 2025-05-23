 /*
  * Copyright (c) Project KanTV. 2021-2023
  *
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
package kantvai.media.player;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import android.os.Build;

import kantvai.media.player.KANTVLog;

public class KANTVLibraryLoader
{
    private static final String TAG = KANTVLibraryLoader.class.getSimpleName();
    private static final String JNILIB_NAME = "kantv-media";
    private static Boolean sLoaded = new Boolean(false);
    private static String sLoadedLibrarySuffix = "";

    private KANTVLibraryLoader() {}

    public static boolean hasLoaded()
    {
        return sLoaded;
    }

    public static void load() throws UnsatisfiedLinkError
    {
        synchronized (sLoaded)
        {
            if (!sLoaded)
            {
                load(JNILIB_NAME);
                sLoaded = true;
            }
        }
    }

    public static void load(String libName) throws UnsatisfiedLinkError
    {
        if (!loadJNILib(libName))
        {
            KANTVLog.j(TAG, "failed load jnilib " + libName);
            throw new UnsatisfiedLinkError();
        }
        sLoaded = true;
    }

    public static String getLoadedLibrarySuffix()
    {
        return sLoadedLibrarySuffix;
    }

    private static String getCPUInfoField( String cpuInfo, String field_name )
    {
        if (cpuInfo == null || field_name == null)
        {
            return null;
        }

        String findStr = "\n" + field_name + "\t: ";
        int stringStart = cpuInfo.indexOf(findStr);
        if (stringStart < 0)
        {
            findStr = "\n" + field_name + ": ";
            stringStart = cpuInfo.indexOf(findStr);
            if (stringStart < 0)
            {
                return null;
            }
        }
        int start = stringStart + findStr.length();
        int end = cpuInfo.indexOf("\n", start);

        return cpuInfo.substring(start, end);
    }

    private static String[] GetCPUSuffixes()
    {
        String cpuInfo = readCPUInfo();
        String cpuArchitecture = getCPUInfoField(cpuInfo,"CPU architecture");
        String cpuFeature = getCPUInfoField(cpuInfo, "Features");
        String vendorId = getCPUInfoField(cpuInfo, "vendor_id");

        String CPU_1 = Build.CPU_ABI;
        String CPU_2 = Build.CPU_ABI2;
        if (!KANTVUtils.getReleaseMode()) {
            KANTVLog.d(TAG, "cpuinfo:\n " +  cpuInfo);
            KANTVLog.j(TAG, "cpuArchitecture  :" + cpuArchitecture);
            KANTVLog.j(TAG, "cpuFeature  :" + cpuFeature);
            KANTVLog.j(TAG, "CPU_ABI  : " + CPU_1);
            KANTVLog.d(TAG, "CPU_ABI2 : " + CPU_2);
        }

        boolean isX86 = false;
        String[] cpus = null;

        if ( vendorId != null &&
                (CPU_1.equals("x86") || CPU_2.equals("x86")) &&
                vendorId.equals("GenuineIntel") )
        {
            cpus = new String[] {"x86"};
            isX86 = true;
        }

        if (!isX86 && cpuArchitecture != null)
        {
            if (cpuArchitecture.startsWith("7") || cpuArchitecture.startsWith("8"))
            {
                if (cpuFeature != null && cpuFeature.contains("neon"))
                {
                    cpus = new String[] {"arm64-v8a", "armv8", "armeabi-v7a", "armv7"};
                }
                else if (cpuArchitecture.startsWith("8"))
                {
                    cpus = new String[] {"arm64-v8a", "armv8", "armeabi-v7a", "armv7"};
                }
            }
            else if (cpuArchitecture.startsWith("6"))
            {
                cpus = new String[] {"armv6", "armv5"};
             }
             else if (cpuArchitecture.startsWith("5"))
             {
                cpus = new String[] {"armv5"};
             }

        }

        return cpus;
    }

    private static String readCPUInfo()
    {
        String result = " ";
        try
        {
            if (new File("/proc/cpuinfo").exists())
            {
                BufferedReader brCpuInfo = new BufferedReader(new FileReader(new File("/proc/cpuinfo")));
                String aLine;

                if (brCpuInfo != null)
                {
                    while ((aLine = brCpuInfo.readLine()) != null)
                    {
                        result = result + aLine + "\n";
                    }
                    brCpuInfo.close();
                }
            }
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }

        return result;
    }

    private static boolean loadJNILib(String lib)
    {
        boolean loaded = false;

        String[] cpus = GetCPUSuffixes();
        List<String> list = new ArrayList<String>();

        for (String cpu: cpus)
        {
            if (android.os.Build.VERSION.SDK_INT < 16 )
            {
                list.add(lib +  "_" + cpu);
            }
            else
            {
                list.add(lib +  "_" + cpu);
            }
        }

        list.add(lib);

        for (String temp : list)
        {
            KANTVLog.d(TAG, "lib : " + temp);
        }

        for (String str : list)
        {
            try
            {
                KANTVLog.d(TAG, "try to loadLibrary " + str);
                System.loadLibrary(str);
                sLoadedLibrarySuffix = str.substring(lib.length());
                //if (!KANTVUtils.getReleaseMode())
                {
                    KANTVLog.j(TAG, "loadLibrary " + str + " succeed");
                }
                loaded = true;
                break;
            }
            catch (UnsatisfiedLinkError ule)
            {
                //if (!KANTVUtils.getReleaseMode())
                {
                    KANTVLog.d(TAG, "can't load " + str + ule.toString());
                }
            }
        }

        return loaded;
    }

    public static native int kantv_anti_remove_rename_this_file();
}