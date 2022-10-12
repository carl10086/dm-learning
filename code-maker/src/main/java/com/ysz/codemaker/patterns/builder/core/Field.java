package com.ysz.codemaker.patterns.builder.core;

/**
 * @author carl
 * @create 2022-10-12 4:38 PM
 **/
public record Field(
    /*filed name*/
    String name,
    /*full class name*/
    String fullClassname,
    /*showClassname*/
    String showClassname,
    /*type*/
    FieldType type
) {


  public static Field ofSimple(
      String name,
      String fullClassname
  ) {
    return new Field(
        name,
        fullClassname,
        fullClassname.substring(fullClassname.lastIndexOf(".") + 1),
        FieldType.simple
    );
  }

  public static Field ofArray(
      String name,
      String fullClassname
  ) {
    return new Field(
        name,
        fullClassname,
        fullClassname.substring(fullClassname.lastIndexOf(".") + 1) + "[]",
        FieldType.array
    );
  }


}
