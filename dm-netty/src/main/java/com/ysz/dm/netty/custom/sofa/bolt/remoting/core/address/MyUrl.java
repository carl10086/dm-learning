package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.address;

import com.alipay.remoting.config.Configs;
import com.alipay.remoting.rpc.protocol.RpcProtocolV2;
import java.util.Properties;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class MyUrl {

  private String originUrl;

  /** ip, can be number format or hostname format*/
  private String ip;

  /** port, should be integer between (0, 65535]*/
  private int port;

  /** unique key of this url */
  private String uniqueKey;

  /** URL args: timeout value when do connect */
  private int connectTimeout;

  /** URL args: protocol */
  private byte protocol;

  /** URL args: version */
  private byte version = RpcProtocolV2.PROTOCOL_VERSION_1;

  /** URL agrs: connection number */
  private int connNum = Configs.DEFAULT_CONN_NUM_PER_URL;

  /** URL agrs: whether need warm up connection */
  private boolean connWarmup;

  /** URL agrs: all parsed args of each originUrl */
  private Properties properties;
}
