package com.ysz.biz.validator;

import java.util.Set;
import javax.validation.ConstraintViolation;
import javax.validation.Path;
import javax.validation.Validation;
import javax.validation.Validator;
import javax.validation.ValidatorFactory;
import javax.validation.metadata.ConstraintDescriptor;
import org.hibernate.validator.HibernateValidator;

/**
 * @author carl
 */
public class Validate_Dm_001 {

  private ValidatorFactory factory;

  private Validator validator;


  public void validateComplexBean() {

    factory = Validation.byProvider(HibernateValidator.class)
        .configure()
        // 快速失败模式
        .failFast(true)
        // .addProperty( "hibernate.validator.fail_fast", "true" )
        .buildValidatorFactory();

    validator = factory.getValidator();
    Set<ConstraintViolation<Object>> validateRes = validator.validate(ComplexBean.mock());
    for (ConstraintViolation<Object> validateRe : validateRes) {
      Path propertyPath = validateRe.getPropertyPath();
      System.out.println(validateRe.getMessage());
    }
  }

  public static void main(String[] args) {
    new Validate_Dm_001().validateComplexBean();
  }

}
