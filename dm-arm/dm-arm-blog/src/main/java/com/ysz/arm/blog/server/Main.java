package com.ysz.arm.blog.server;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/15
 **/
@Slf4j
@SpringBootApplication(
    exclude = {
//        ArmeriaAutoConfiguration.class
    }
)
public class Main {

  public static void main(String[] args) throws Exception {
    SpringApplication.run(Main.class, args);
  }
}
