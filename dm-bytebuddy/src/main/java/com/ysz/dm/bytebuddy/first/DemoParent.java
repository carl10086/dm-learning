package com.ysz.dm.bytebuddy.first;

import java.util.Date;
import lombok.Data;

/**
 * @author carl
 */
@Data
public class DemoParent {


  private String name = "carl";

  private Integer age = 11;

  private Date createAt = new Date();


}
