package com.ysz.arm.blog.server.core.exceptions;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/16
 **/
public class CustomInvalidException extends RuntimeException {

  private static final long serialVersionUID = 2150479758497151432L;

  public CustomInvalidException() {
    super();
  }

  public CustomInvalidException(String message) {
    super(message);
  }

  public CustomInvalidException(String message, Throwable cause) {
    super(message, cause);
  }

  public CustomInvalidException(Throwable cause) {
    super(cause);
  }

  protected CustomInvalidException(
      String message,
      Throwable cause,
      boolean enableSuppression,
      boolean writableStackTrace
  ) {
    super(message, cause, enableSuppression, writableStackTrace);
  }
}
