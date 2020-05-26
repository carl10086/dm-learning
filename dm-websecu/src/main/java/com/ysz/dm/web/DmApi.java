package com.ysz.dm.web;

import javax.annotation.Resource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author carl.yu
 * @date 2020/3/18
 */
@RequestMapping("/dm")
@RestController
public class DmApi {

  @Resource
  private FeedRecRedisProperties feedRecRedisProperties;

  @GetMapping("/")
  public String getData() {
    System.err.println(feedRecRedisProperties);
    return "hello world";
  }

}
