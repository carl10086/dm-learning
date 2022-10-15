package com.ysz.arm.blog.server.config;

import com.linecorp.armeria.common.RequestContext;
import java.util.Arrays;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/16
 **/
@Aspect
@Slf4j
public class GrpcServiceAspect {

  @Pointcut("execution(* (@com.ysz.arm.blog.server.config.ArmeriaGrpc *).*(..))")
  public void rpcAutoExceptionHandlerAnnoClassMethod() {
  }

  @Around("rpcAutoExceptionHandlerAnnoClassMethod()")
  public Object invoke(ProceedingJoinPoint joinPoint) {
    Object ret = null;
    if (log.isDebugEnabled()) {
      var ctx = new Ctx(
          joinPoint.getSignature().toString(),
          Arrays.toString(joinPoint.getArgs())
      );

      RequestContext requestContext = RequestContext.currentOrNull();
      log.debug("request ctx:{}", requestContext);
      log.debug("ctx:{}", ctx);
    }
    try {
      ret = joinPoint.proceed();
    } catch (Throwable e) {
      log.error("errors", e);
    }
    return ret;
  }


  public static record Ctx(
      String methodInfo,
      String argsInfo) {

  }
}
