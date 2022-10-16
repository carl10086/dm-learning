package com.ysz.arm.blog.server.core.exceptions;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/16
 **/
public class CustomServerErrorException extends RuntimeException {

  private static final long serialVersionUID = -5422952006301425870L;

  public CustomServerErrorException() {
    super();
  }

  public CustomServerErrorException(String message) {
    super(message);
  }

  public CustomServerErrorException(String message, Throwable cause) {
    super(message, cause);
  }

  public CustomServerErrorException(Throwable cause) {
    super(cause);
  }

  protected CustomServerErrorException(
      String message,
      Throwable cause,
      boolean enableSuppression,
      boolean writableStackTrace
  ) {
    super(message, cause, enableSuppression, writableStackTrace);
  }
}
