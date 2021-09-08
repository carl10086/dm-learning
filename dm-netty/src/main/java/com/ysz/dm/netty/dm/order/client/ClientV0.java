package com.ysz.dm.netty.dm.order.client;

import com.ysz.dm.netty.dm.order.Constants;
import com.ysz.dm.netty.dm.order.client.codec.OrderCliFrameDecoder;
import com.ysz.dm.netty.dm.order.client.codec.OrderCliFrameEncoder;
import com.ysz.dm.netty.dm.order.client.codec.OrderCliProtocolDecoder;
import com.ysz.dm.netty.dm.order.client.codec.OrderCliProtocolEncoder;
import com.ysz.dm.netty.dm.order.domain.MessageReq;
import com.ysz.dm.netty.dm.order.domain.order.OrderOperation;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;

/**
 * @author carl
 */
public class ClientV0 {

  public static void main(String[] args) throws Exception {
    EventLoopGroup workerGroup = new NioEventLoopGroup();
    try {
      Bootstrap bootstrap = new Bootstrap();

      bootstrap.group(workerGroup)
          .channel(NioSocketChannel.class)
          .option(ChannelOption.TCP_NODELAY, true)
          .handler(new ChannelInitializer<SocketChannel>() {
            @Override
            protected void initChannel(SocketChannel ch) throws Exception {
              ChannelPipeline pipeline = ch.pipeline();
              pipeline.addLast(new OrderCliFrameDecoder());
              pipeline.addLast(new OrderCliProtocolDecoder());
              pipeline.addLast(new OrderCliFrameEncoder());
              pipeline.addLast(new OrderCliProtocolEncoder());

              pipeline.addLast(new LoggingHandler(LogLevel.INFO));
            }
          });

      ChannelFuture channelFuture = bootstrap.connect("127.0.0.1", Constants.PORT);
      channelFuture.sync();

      MessageReq req = MessageReq.of(1L, new OrderOperation(1000, "tudou"));

      channelFuture.channel().writeAndFlush(req);
      channelFuture.channel().closeFuture().sync();
    } finally {
      workerGroup.shutdownGracefully();
    }


  }

}
