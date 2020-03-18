package com.ysz.dm.web;

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

  @GetMapping("/")
  public String getData() {
    return "hello world";
  }

}
