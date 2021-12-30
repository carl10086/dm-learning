package com.ysz.dm.dubbo3.demo.provider;

import org.springframework.context.annotation.AnnotationConfigApplicationContext;


public class ProviderApp {

  public static void main(String[] args) throws Exception {
    System.setProperty("dubbo.application.logger", "slf4j");
//    ClassPathXmlApplicationContext context = new ClassPathXmlApplicationContext("classpath:dubbo-demo-provider.xml");
    AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(DubboConfig.class);
    context.start();
    System.in.read();
  }

}
