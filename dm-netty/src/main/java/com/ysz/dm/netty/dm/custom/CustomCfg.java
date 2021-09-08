package com.ysz.dm.netty.dm.custom;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@ToString
public class CustomCfg {

  /**
   * 封装 tcp 相关参数
   */
  @Getter
  @Setter
  private CustomTcpCfg tcpCfg;

  /**
   * 分装 netty 相关参数
   */
  @Getter
  @Setter
  private CustomNettyCfg nettyCfg;

  public Integer tcp_so_backlog() {
    return tcpCfg.getBacklog();
  }

  public Boolean tcp_so_reuseaddr() {
    return tcpCfg.getReuseAddr();
  }

  public Boolean tcp_nodelay() {
    return tcpCfg.getNoDelay();
  }

  public Boolean tcp_so_keepalive() {
    return tcpCfg.getKeepalive();
  }

  public int netty_buffer_low_watermark() {
    return this.nettyCfg.getLowWaterMark();
  }

  public int netty_buffer_high_watermark() {
    return this.nettyCfg.getHighWaterMark();
  }


  @Data
  private static class CustomTcpCfg {

    /**
     * tcp send buf 参数、单位字节
     * 建议为空，可以使用 linux 的自动调节机制
     */
    private Integer soSndBuf = null;

    /**
     * tcp rev buf 参数、单位字节
     * 建议为空，可以使用 linux 的自动调节机制
     */
    private Integer soRcvBuf = null;

    /**
     * tcp no delay 参数 . 延迟敏感类 可以设置 nodelay
     */
    private Boolean noDelay = true;

    /**
     * 默认 backlog 队列 = 1024
     */
    private Integer backlog = 1024;

    /**
     * 默认 keepalive = true
     */
    private Boolean keepalive = true;

    /**
     * 默认 reuseAddr = true
     */
    private Boolean reuseAddr = true;
  }

  @Data
  private static class CustomNettyCfg {

    private Integer lowWaterMark = 32 * 1024;

    private Integer highWaterMark = 64 * 1024;

  }
}
