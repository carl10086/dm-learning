package com.ysz.dm.lib.validator;

import java.util.Objects;
import java.util.Set;
import javax.validation.ConstraintViolation;
import javax.validation.Validation;
import javax.validation.Validator;
import javax.validation.ValidatorFactory;
import org.hibernate.validator.HibernateValidator;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
public class HibernateValidateTools {

  private static ValidatorFactory factory;
  private static Validator validator;

  static {
    factory =
        Validation.byProvider(HibernateValidator.class)
            .configure()
            // 快速失败模式
            .failFast(true)
            // .addProperty( "hibernate.validator.fail_fast", "true" )
            .buildValidatorFactory();
    validator = factory.getValidator();
  }


  public static void chkAndThrow(Object bean) throws RuntimeException {
    validator = factory.getValidator();
    Set<ConstraintViolation<Object>> validateRes = validator.validate(bean);
//    if (CollectionTools.isNotEmpty(validateRes)) {
    if (null != validateRes && validateRes.size() > 0) {

      ConstraintViolation<Object> first = validateRes.iterator().next();
      if (first != null) {
        String path = Objects.toString(first.getPropertyPath());
        String message = first.getMessage();
        throw new IllegalArgumentException(path + ":" + message);
      }
    }
  }


}
