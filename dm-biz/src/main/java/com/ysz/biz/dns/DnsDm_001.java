package com.ysz.biz.dns;

import java.io.IOException;
import org.xbill.DNS.DClass;
import org.xbill.DNS.Flags;
import org.xbill.DNS.Message;
import org.xbill.DNS.Name;
import org.xbill.DNS.Rcode;
import org.xbill.DNS.Record;
import org.xbill.DNS.Resolver;
import org.xbill.DNS.SimpleResolver;
import org.xbill.DNS.Type;

public class DnsDm_001 {

  public static void main(String[] args) throws Exception {
    SimpleResolver sr = new SimpleResolver("114.114.114.114");
    System.out.println("Standard resolver:");
    sendAndPrint(sr, "www.baidu.com");
  }


  private static void sendAndPrint(Resolver vr, String name) throws IOException {
    System.out.println("\n---" + name);
    Record qr = Record.newRecord(Name.fromConstantString(name), Type.A, DClass.IN);
    Message response = vr.send(Message.newQuery(qr));
    System.out.println("AD-Flag: " + response.getHeader().getFlag(Flags.AD));
    System.out.println("RCode:   " + Rcode.string(response.getRcode()));
  }
}
