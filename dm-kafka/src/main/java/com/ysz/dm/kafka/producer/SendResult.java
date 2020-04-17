package com.ysz.dm.kafka.producer;

import lombok.Getter;

/**
 * 消息发送结果, 实际上底层框架的包装类。暂时只有调试的时候打印功能
 *
 * @author carl.yu
 * @date 2018/9/17
 */
@Getter
public class SendResult {


  /**
   * 发送是否成功
   */
  private final boolean success;

  /**
   * 失败包装错误
   */
  private final Exception error;

  /**
   * 错误输出, 方便输出
   */
  private final String errorMsg;


  public final Object reference;


  private SendResult(boolean success, Exception error, String errorMsg, Object reference) {
    this.success = success;
    this.error = error;
    this.errorMsg = errorMsg;
    this.reference = reference;
  }

  public static SendResult success(Object reference) {
    return new SendResult(true, null, null, reference);
  }

  public static SendResult error(Exception error) {
    return new SendResult(false, error, null, null);
  }

  public static SendResult error(Exception error, String errorMsg) {
    return new SendResult(false, error, errorMsg, null);
  }

  @Override
  public String toString() {
    return "SendResult{" +
        "success=" + success +
        ", error=" + error +
        ", errorMsg='" + errorMsg + '\'' +
        ", reference=" + reference +
        '}';
  }
}
