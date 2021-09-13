package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.command;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.MyProtocolCode;
import java.io.Serializable;

public interface MyRemotingCommand extends Serializable {

  MyProtocolCode getProtocolCode();




}
