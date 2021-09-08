package com.ysz.dm.netty.custom.sofa.bolt.remoting.server;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltGenericOptions;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltOption;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltServerOptions;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.address.MyAddressParser;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.address.MyAddressParserImpl;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.MyConnectionManager;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.impl.MyDefaultConnectionManager;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.impl.MyRandomSelectStrategy;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.infra.thread.MyNamedThreadFactory;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.infra.utils.MyNettyEventLoopUtils;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;

public class MyRpcServer extends MyAbstractRemotingServer {

  private ServerBootstrap bootstrap;
  private ChannelFuture channelFuture;
  private MyAddressParser myAddressParser;
  private MyConnectionManager connectionManager;


  private final EventLoopGroup boss = MyNettyEventLoopUtils.newEventLoopGroup(
      1,
      new MyNamedThreadFactory("Rpc-netty-server-boss", false)
  );

  private final EventLoopGroup worker = MyNettyEventLoopUtils.newEventLoopGroup(
      Runtime.getRuntime().availableProcessors() * 2,
      new MyNamedThreadFactory("Rpc-netty-server-worker", true)
  );


  public MyRpcServer(
      String ip,
      int port) {
    this(ip, port, false, false);
  }

  @Override
  protected void doInit() {
    initAddressParser();

    if (getOption(MyBoltServerOptions.SERVER_MANAGE_CONNECTION_SWITCH)) {
      final MyRandomSelectStrategy connectionSelectStrategy = new MyRandomSelectStrategy(
          super.options);
      this.connectionManager = new MyDefaultConnectionManager();
    } else {

    }

    this.bootstrap = new ServerBootstrap();
    this.bootstrap.group(boss, worker)
        .channel(MyNettyEventLoopUtils.getServerSocketChannelClass())
        .option(ChannelOption.SO_BACKLOG, getOption(MyBoltServerOptions.TCP_SO_BACKLOG))
        .option(ChannelOption.SO_REUSEADDR, getOption(MyBoltGenericOptions.TCP_SO_REUSEADDR))


  }

  private void initAddressParser() {
    if (this.myAddressParser == null) {
      this.myAddressParser = new MyAddressParserImpl();
    }
  }

  public MyRpcServer(
      String ip,
      int port,
      boolean manageConnection,
      boolean syncStop
  ) {
    super(ip, port);
    super
        .updateOption(MyBoltServerOptions.SERVER_MANAGE_CONNECTION_SWITCH, manageConnection)
        .updateOption(MyBoltServerOptions.SERVER_SYNC_STOP, syncStop);
  }

}
