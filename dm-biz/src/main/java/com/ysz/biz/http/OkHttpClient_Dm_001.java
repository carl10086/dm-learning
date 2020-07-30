package com.ysz.biz.http;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
import lombok.extern.slf4j.Slf4j;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.apache.commons.io.IOUtils;
import org.jetbrains.annotations.NotNull;

/**
 * @author carl
 */
public class OkHttpClient_Dm_001 {

  private final ObjectMapper objectMapper = new ObjectMapper();

  public final OkHttpClient okHttpClient = new OkHttpClient.Builder()
      .addInterceptor(new LoggingInterceptor())
      .readTimeout(10L, TimeUnit.MILLISECONDS)
      .build();

  public void tstGet() {

    final Request request = new Request.Builder()
        .url("https://c-ssl.duitang.com/uploads/item/202005/14/20200514123116_ctfff.png")
        .build();

    try (Response response = okHttpClient.newBuilder().readTimeout(200L, TimeUnit.MILLISECONDS)
        .build()
        .newCall(request).execute()) {
      if (!response.isSuccessful()) {
        System.err.println("请求失败!");
        return;
      }

      final ResponseBody body = response.body();
      assert body != null;
      System.out.println("请求的长度是:" + body.contentLength());
      try (FileOutputStream outputStream = new FileOutputStream("/Users/carl/tmp/image/1.png")) {
        IOUtils.copyLarge(body.byteStream(), outputStream);
      } catch (Exception e) {
        e.printStackTrace();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  public static void main(String[] args) throws Exception {
    new OkHttpClient_Dm_001().tstGet();
  }


  @Slf4j
  private static class LoggingInterceptor implements Interceptor {

    @NotNull
    @Override
    public Response intercept(@NotNull Chain chain) throws IOException {
      Request request = chain.request();

      long t1 = System.nanoTime();
      final String url = request.url().toString();
      log.info(String
          .format("Sending request %s on %s%n%s", url, chain.connection(), request.headers()));
      try {
        return chain.proceed(request);
      } catch (Exception e) {
        System.out.println("吃掉异常");
        throw e;
      } finally {
        long t2 = System.nanoTime();
        log.info(String.format("Received response for %s in %.1fms%n", url, (t2 - t1) / 1e6d));
      }
    }
  }
}
