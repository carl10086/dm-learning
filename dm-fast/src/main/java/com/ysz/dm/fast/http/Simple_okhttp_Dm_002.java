package com.ysz.dm.fast.http;

import okhttp3.OkHttpClient;
import okhttp3.Request.Builder;

/**
 * @author carl
 */
public class Simple_okhttp_Dm_002 {

  public static void main(String[] args) throws Exception {
    OkHttpClient okHttpClient = new OkHttpClient();
    String string = okHttpClient.newCall(new Builder()
        .url("http://www.163.com")
        .get()
        .build()).execute().body().string();
    System.out.println(string);
  }

}
