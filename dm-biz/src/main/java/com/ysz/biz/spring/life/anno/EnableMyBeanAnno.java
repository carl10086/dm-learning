package com.ysz.biz.spring.life.anno;

import com.ysz.biz.spring.life.EnableMyBeanRegistrar;
import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.springframework.context.annotation.Import;

@Target({ElementType.TYPE, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Import(EnableMyBeanRegistrar.class)
public @interface EnableMyBeanAnno {

}
