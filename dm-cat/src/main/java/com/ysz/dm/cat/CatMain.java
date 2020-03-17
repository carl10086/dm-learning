package com.ysz.dm.cat;

import com.dianping.cat.Cat;
import com.dianping.cat.message.Transaction;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
public class CatMain {


  public static void main(String[] args) throws Exception {
    Transaction t = Cat.newTransaction("REPO", "m1");
    t.setStatus(Transaction.SUCCESS);
    t.complete();
  }

}
