//
// (c) 2004-2015 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
package com.konakartadmin.apiexamples;

import java.math.BigDecimal;
import java.util.Date;

import com.konakartadmin.app.AdminCategory;
import com.konakartadmin.app.AdminProduct;
import com.konakartadmin.app.AdminProductAttribute;
import com.konakartadmin.app.AdminProductDescription;
import com.konakartadmin.app.KKAdminException;
import com.konakartadmin.bl.KonakartAdminConstants;

/**
 * This class shows how to call the KonaKart Admin API to insert, edit and then delete a product.
 * Before running you may have to edit BaseApiExample.java to change the username and password used
 * to log into the engine. The default values are admin@konakart.com / princess .
 */
public class ProductCrud extends BaseApiExample
{
    private static final String usage = "Usage: ProductCrud\n" + COMMON_USAGE;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        try
        {
            /*
             * Parse the command line arguments
             */
            parseArgs(args, usage, 0);
            
            /*
             * Get an instance of the Admin KonaKart engine and login. The method called can be
             * found in BaseApiExample.java
             */
            init(getEngineMode(), getStoreId(), getEngClassName(), isCustomersShared(),
                    isProductsShared(), isCategoriesShared(), null);

            /*
             * First let's insert a product
             */
            int prodId = insertProduct();

            /*
             * Now let's read the product from the DB
             */
            AdminProduct prod = eng.getProduct(sessionId, prodId);

            /*
             * Edit the product
             */
            editProduct(prod);

            /*
             * Delete the product
             */
            eng.deleteProduct(sessionId, prodId);

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Edit a product
     * 
     * @param prod
     * @throws KKAdminException
     */
    private static void editProduct(AdminProduct prod) throws KKAdminException
    {
        // Change the type to digital download
        prod.setType(KonakartAdminConstants.DIGITAL_DOWNLOAD_PROD_TYPE);

        // Add a digital download file
        prod.setMaxNumDownloads(-1);
        prod.setMaxDownloadDays(-1);
        prod.setContentType("image/jpeg");
        prod.setFilePath("/downloads/my_download.jpg");

        // Change one of the images
        prod.setImage("test/newImg.gif");

        // Change the quantity
        prod.setQuantity(99);

        // Change a description
        if (prod.getDescriptions() != null && prod.getDescriptions().length > 0)
        {
            prod.getDescriptions()[0].setDescription("This is a new description");
        }

        // Change a category assuming one exists with id = 2
        AdminCategory[] cats = prod.getCategories();
        if (cats != null && cats.length > 0)
        {
            cats[0].setId(2);
        }

        // Remove all attributes
        prod.setAttributes(null);

        eng.editProduct(sessionId, prod);
    }

    /**
     * Insert a new product
     * 
     * @return Returns the numeric id of the inserted product
     * @throws KKAdminException
     */
    private static int insertProduct() throws KKAdminException
    {
        /*
         * Instantiate an AdminProduct object and set some of its attributes
         */
        AdminProduct prod = new AdminProduct();
        // Set the SKU
        prod.setSku("123-abc-456");
        // Assuming we have a manufacturer in the DB with id==1
        prod.setManufacturerId(1);
        // Image that will be used in the application for the product
        prod.setImage("test/test.gif");
        // Product model
        prod.setModel("test model");
        // Product price before applying tax
        prod.setPriceExTax(new BigDecimal(55.99));
        // Quantity of product in stock
        prod.setQuantity(20);
        // Product status: 1=active, 0=inactive
        prod.setStatus((byte) 1);
        // The tax class of the product. Used to determine how much tax to apply.
        prod.setTaxClassId(1);
        // Weight of product. Can be used to calculate shipping cost.
        prod.setWeight(new BigDecimal(5));
        // Defines when the product will be available
        prod.setDateAvailable(new Date());
        // Custom attribute. Can contain custom data.
        prod.setCustom1("custom1");
        // Custom attribute. Can contain custom data.
        prod.setCustom2("custom2");
        // Custom attribute. Can contain custom data.
        prod.setCustom3("custom3");
        // Custom attribute. Can contain custom data.
        prod.setCustom4("custom4");
        // Custom attribute. Can contain custom data.
        prod.setCustom5("custom5");

        /*
         * An Admin Product object has an array of AdminProductDescription objects, each one of
         * which contains a name, description and url for a different language. The url is language
         * dependent since it can point to the home page of the product for the country in question.
         */
        AdminProductDescription[] descriptions = new AdminProductDescription[2];
        descriptions[0] = new AdminProductDescription();
        descriptions[0].setDescription("Test prod - English");
        descriptions[0].setLanguageId(1);
        descriptions[0].setName("Test Prod - E");
        descriptions[0].setUrl("www.testprod.co.uk");
        descriptions[1] = new AdminProductDescription();
        descriptions[1].setDescription("Test prod - German");
        descriptions[1].setLanguageId(2);
        descriptions[1].setName("Test Prod - D");
        descriptions[1].setUrl("www.testprod.de");
        prod.setDescriptions(descriptions);

        /*
         * An Admin Product may belong to one or more categories. We create an array of categories
         * and add one category to the array which we then add to the product.
         */
        AdminCategory cat = new AdminCategory();
        // Assuming we have a category with id==1 in the database
        cat.setId(1);
        AdminCategory[] catArray = new AdminCategory[1];
        catArray[0] = cat;
        prod.setCategories(catArray);

        /*
         * The following Admin Product Attributes are optional. They are used if the product has
         * options which can be selected to configure the product. These options may add or subtract
         * a value from the final price. i.e. Size "extra small" may be -$10.00 from the product
         * price while size "extra large" may be +$10.00.
         */

        // We will add two attributes
        AdminProductAttribute[] attrs = new AdminProductAttribute[2];

        // Attribute 1
        AdminProductAttribute prodAttr1 = new AdminProductAttribute();
        // This points to the option in the products_options table with id = 4 (Memory in demo
        // DB)
        prodAttr1.setOptionId(4);
        // This points to the option value in the products_options_values table with id = 1 (4MB
        // in demo DB)
        prodAttr1.setOptionValueId(1);
        // This option costs an extra 20 if selected
        prodAttr1.setPrice(new BigDecimal(20));
        prodAttr1.setPricePrefix("+");
        attrs[0] = prodAttr1;

        // Attribute 2
        AdminProductAttribute prodAttr2 = new AdminProductAttribute();
        // This points to the option in the products_options table with id = 3 (Model in demo
        // DB)
        prodAttr2.setOptionId(3);
        // This points to the option value in the products_options_values table with id = 5
        // (Value model
        // in demo DB)
        prodAttr2.setOptionValueId(5);
        // The product price is reduced by 10 if this option is selected
        prodAttr2.setPrice(new BigDecimal(10));
        prodAttr2.setPricePrefix("-");
        attrs[1] = prodAttr2;

        // Add the attributes to the product
        prod.setAttributes(attrs);

        // Insert the product and get the product Id
        int prodId = eng.insertProduct(sessionId, prod);

        System.out.println("Product Id of inserted product = " + prodId);

        return prodId;
    }
}