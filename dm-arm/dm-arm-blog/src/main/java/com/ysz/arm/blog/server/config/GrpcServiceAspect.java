package com.ysz.arm.blog.server.config;

import com.google.rpc.Code;
import com.linecorp.armeria.common.RequestContext;
import com.ysz.arm.blog.server.core.exceptions.CustomInvalidException;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.StreamObserver;
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

      StreamObserver<?> observer = findStreamObserverFromArgs(joinPoint);
      if (observer == null) {
        log.error("null observer is found , why ?", e);
      } else {
        if (e instanceof CustomInvalidException) {
          CustomInvalidException invalidException = (CustomInvalidException) e;
          log.warn("param exception, e");

          com.google.rpc.Status status = com.google.rpc.Status.newBuilder()
              .setCode(Code.INVALID_ARGUMENT.getNumber())
              .setMessage(invalidException.getMessage())
              .build();
          observer.onError(StatusProto.toStatusRuntimeException(status));
        }
      }
    }
    return ret;
  }


  private StreamObserver<?> findStreamObserverFromArgs(ProceedingJoinPoint joinPoint) {
    Object[] args = joinPoint.getArgs();

    if (null == args || args.length == 0) {
      return null;
    }

    for (int i = args.length - 1; i > 0; i--) {
      Object arg = args[i];
      if (arg instanceof StreamObserver) {
        return (StreamObserver<?>) arg;
      }
    }

    return null;
  }


  public static record Ctx(
      String methodInfo,
      String argsInfo) {

  }
}
