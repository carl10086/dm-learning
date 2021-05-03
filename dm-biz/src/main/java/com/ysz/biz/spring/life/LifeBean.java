package com.ysz.biz.spring.life;

import javax.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.context.SmartLifecycle;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class LifeBean implements SmartLifecycle, InitializingBean {

  public LifeBean(){
    log.info("LifeBean constructed");
  }

  private volatile boolean running = false;

  public void sayHi(String name) {
    log.info("LifeBean sayHi:{}", name);
  }

  /**
   * 1. 我们主要在该方法中启动任务或者其他异步服务，比如开启MQ接收消息<br/>
   * 2. 当上下文被刷新（所有对象已被实例化和初始化之后）时，将调用该方法，默认生命周期处理器将检查每个SmartLifecycle对象的isAutoStartup()方法返回的布尔值。
   * 如果为“true”，则该方法会被调用，而不是等待显式调用自己的start()方法。
   */
  @Override
  public void start() {
    log.info("When application start");
    this.running = true;
  }

  @Override
  public void stop() {
    log.info("When application end");
    this.running = false;
  }

  /**
   * 根据该方法的返回值决定是否执行start方法。<br/>
   * 返回true时start方法会被自动执行，返回false则不会。
   */
//  @Override
//  public boolean isAutoStartup() {
//    return true;
//  }

  /**
   * 1. 只有该方法返回false时，start方法才会被执行。<br/>
   * 2. 只有该方法返回true时，stop(Runnable callback)或stop()方法才会被执行。
   */
  @Override
  public boolean isRunning() {
    return this.running;
  }


  @Override
  public int getPhase() {
    return 0;
  }

  @PostConstruct
  public void postConstruct() {
    log.info("@postConstruct");
  }

  @Override
  public void afterPropertiesSet() throws Exception {
    log.info("afterPropertiesSet");
  }
}
