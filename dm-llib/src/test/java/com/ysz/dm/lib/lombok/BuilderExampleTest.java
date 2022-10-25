package com.ysz.dm.lib.lombok;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

/**
 * @author carl
 * @create 2022-10-25 6:57 PM
 **/
@Slf4j
public class BuilderExampleTest {


  @Test
  public void testBuild() {
    var example = BuilderExample.builder()
        .occupation("1") // add single value
        .occupation("2")
        .build();

    log.info("example:{}", example);

  }
}