package com.ysz.biz;

import com.ysz.biz.http.OkHttpClient_Dm_001;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.apache.commons.lang3.RegExUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.PatternMatchUtils;

public class Tmp {

  static final OkHttpClient okHttpClient = new OkHttpClient.Builder()
      .readTimeout(10L, TimeUnit.MILLISECONDS)
      .build();

  public static void check() {
//    final Request request = new Request.Builder()
//        .url("http://192.168.3.64:3000/health")
//        .build();
    final Request request = new Request.Builder()
        .url("http://127.0.0.1:1234/health")
        .build();
    try (Response response = okHttpClient.newBuilder().readTimeout(200L, TimeUnit.MILLISECONDS)
        .build()
        .newCall(request).execute()) {

      if (response.isSuccessful()) {
        final String res = response.body().string();
        if ("ok".equalsIgnoreCase(res)) {
          System.out.println("ok");
        } else {
          System.err.println("fail");
        }
      } else {
        System.err.println("fail");
      }

    } catch (Exception e) {
      System.err.println("fail");
    }
  }


  public static void main(String[] args) throws Exception {

    for (; ; ) {
      check();
      Thread.sleep(1000L);
    }

  }

}
