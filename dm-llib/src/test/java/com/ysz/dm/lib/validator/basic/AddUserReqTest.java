package com.ysz.dm.lib.validator.basic;

import com.ysz.dm.lib.validator.HibernateValidateTools;
import org.junit.Assert;
import org.junit.Test;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
public class AddUserReqTest {


  @Test
  public void testUsernameNotNull() {
    var message = "username:username can't be null";
    Assert.assertThrows(message,
                        IllegalArgumentException.class,
                        () -> HibernateValidateTools.chkAndThrow(new AddUserReq(null, null))
    );
  }


  @Test
  public void testPasswordLength() {
    var message = "password:Password must be 8 or more characters in length.";
    Assert.assertThrows(message,
                        IllegalArgumentException.class,
                        () -> HibernateValidateTools.chkAndThrow(new AddUserReq("aaa", "123"))
    );
  }
}