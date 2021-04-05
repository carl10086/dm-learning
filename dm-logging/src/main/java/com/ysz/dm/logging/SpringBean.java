package com.ysz.dm.logging;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class SpringBean {

  @Value("${spring.profiles.active}")
  private String profile;

  public void hello() {
    System.err.println("profile:" + profile);
    log.error("hello world, {}, {}", "1", "2", "3", "4", new RuntimeException());
  }

}
