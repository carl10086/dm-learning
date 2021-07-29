package com.ysz.biz.zk.dubbo;

import com.google.common.base.Splitter;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.retry.ExponentialBackoffRetry;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class DubboDm {

  private CuratorFramework curatorFramework;

  @Before
  public void setUp() throws Exception {
    curatorFramework = CuratorFrameworkFactory
        .newClient("10.1.5.72:3881", new ExponentialBackoffRetry(1000, 3));
    curatorFramework.start();
  }

  @Test
  public void tstUrl() throws Exception {
    String urlStr = "consumer://10.1.5.113/org.apache.dubbo.rpc.service.GenericService?application=k2d-consumer&category=consumers&check=false&dubbo=2.0.2&generic=true&group=saturn-offline&interface=com.duitang.saturn.client.user.service.IUserService&metadata-type=remote&pid=29877&release=2.7.10&side=consumer&sticky=false&timeout=5000&timestamp=1620376915035&version=1.0";
    final String interfaceName = "org.apache.dubbo.rpc.service.GenericService";


  }

  private String decode(String value) {
    try {
      return URLDecoder.decode(value, StandardCharsets.UTF_8.toString());
    } catch (UnsupportedEncodingException e) {
      e.printStackTrace();
      return null;
    }
  }


  @Test
  public void tst() throws Exception {
    final String interfaceName = "com.duitang.munich.client.tip.service.ITipService";
    final String consumersPath = String.format("/dubbo/%s/providers", interfaceName);

    List<MyUrl> urls = curatorFramework.getChildren().forPath(consumersPath).stream()
        .map(x -> new MyUrl(decode(x), interfaceName)).collect(
            Collectors.toList());

    urls.forEach(
        x -> System.err.println(x.asStr())
    );

  }


  @After
  public void destory() {
    if (curatorFramework != null) {
      curatorFramework.close();
    }
  }

  private static class MyUrl {

    private final Map<String, String> data;

    public String application() {
      return data.get("application");
    }

    public String pid() {
      return data.get("pid");
    }

    public String group() {
      return data.get("group");
    }

    public String asStr() {
      return new StringBuilder()
          .append("pid:").append(pid()).append(" <->  ")
          .append("application:").append(application()).append(" <->  ")
          .append("group:").append(group()).append(" <->  ")
          .toString();
    }

    private MyUrl(String urlStr, String interfaceName) {
      int start = urlStr.indexOf(interfaceName) + interfaceName.length() + 1;
      List<String> strings = Splitter.on("&").trimResults().splitToList(urlStr.substring(start));
      this.data = new HashMap<>(strings.size());
      for (String string : strings) {
        String[] kv = string.split("=", -1);
        this.data.put(kv[0], kv[1]);
      }
    }
  }

}
