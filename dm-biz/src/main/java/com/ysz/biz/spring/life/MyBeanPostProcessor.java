package com.ysz.biz.spring.life;

import lombok.extern.slf4j.Slf4j;
import org.springframework.aop.support.AopUtils;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class MyBeanPostProcessor implements BeanPostProcessor {

  public MyBeanPostProcessor() {
    log.info("MyBeanPostProcessor constructed");
  }

  @Override
  public Object postProcessBeforeInitialization(Object bean, String beanName)
      throws BeansException {
    if (AopUtils.getTargetClass(bean) == LifeBean.class) {
      log.info("postProcessBeforeInitialization:{}", beanName);
    }
    return bean;
  }

  @Override
  public Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {
    if (AopUtils.getTargetClass(bean) == LifeBean.class) {
      log.info("postProcessAfterInitialization:{}", beanName);
    }
    return bean;
  }
}
