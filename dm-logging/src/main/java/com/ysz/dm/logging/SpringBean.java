package com.ysz.dm.logging;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class SpringBean {

  public void hello() {
    log.error("hello world");
  }

}
