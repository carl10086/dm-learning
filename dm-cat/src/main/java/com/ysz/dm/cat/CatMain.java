package com.ysz.dm.cat;

import com.dianping.cat.Cat;
import com.dianping.cat.message.Transaction;
import java.util.HashMap;
import java.util.Map;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
public class CatMain {

  static class DubboCatContext implements Cat.Context {

    private Map<String, String> properties = new HashMap<String, String>();

    @Override
    public void addProperty(String key, String value) {
      properties.put(key, value);
    }

    @Override
    public String getProperty(String key) {
      return properties.get(key);
    }
  }


  public static void outter() {
    Transaction t = Cat.newTransaction("REPO", "outter");
    inner();
//    DubboCatContext catContext = new DubboCatContext();
//    Cat.logRemoteCallClient(catContext);
    t.setStatus(Transaction.SUCCESS);
    t.complete();
//    System.out.println(catContext.properties);
  }

  public static void inner() {
    Transaction t = Cat.newTransaction("REPO", "inner");
    t.setStatus(Transaction.SUCCESS);

    t.complete();
  }

  public static void main(String[] args) throws Exception {
    outter();
    System.in.read();
  }

}
