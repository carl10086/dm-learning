package com.ysz.arm.blog.server.monitor;

import lombok.extern.slf4j.Slf4j;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/16
 **/
@Slf4j
public class MonitorCtx implements AutoCloseable {

  @Override
  public void close() throws Exception {
    log.info("close ctx{}", this);
  }
}
