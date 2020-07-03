package com.ysz.biz.validator;

import javax.validation.constraints.Min;
import javax.validation.constraints.NotNull;
import lombok.Data;

/**
 * @author carl
 */
@Data
public class ComplexBean {

  @NotNull
  private String username;

  @NotNull
  @Min(value = 10L)
  private Integer age;


  public static ComplexBean mock() {
    ComplexBean complexBean = new ComplexBean();
    complexBean.setUsername("carl");
    complexBean.setAge(1);
    return complexBean;
  }
}
