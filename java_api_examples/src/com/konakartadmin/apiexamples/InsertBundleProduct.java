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
import com.konakartadmin.app.AdminDataDescriptor;
import com.konakartadmin.app.AdminProduct;
import com.konakartadmin.app.AdminProductAttribute;
import com.konakartadmin.app.AdminProductDescription;
import com.konakartadmin.app.AdminProducts;
import com.konakartadmin.app.KKAdminException;
import com.konakartadmin.bl.KonakartAdminConstants;

/**
 * This class shows how to call the KonaKart Admin API to insert a bundle product into the database.
 * Before running you may have to edit BaseApiExample.java to change the username and password used
 * to log into the engine. The default values are admin@konakart.com / princess .
 */
public class InsertBundleProduct extends BaseApiExample
{
    private static final String usage = "Usage: InsertProduct\n" + COMMON_USAGE;

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

            // Insert a simple product
            AdminProduct prod1 = insertProduct("prod1", KonakartAdminConstants.PHYSICAL_PROD_TYPE, /* bundledProds */
                    null, /* addOptions */false);

            // Insert a product with options
            AdminProduct prod2 = insertProduct("prod2", KonakartAdminConstants.PHYSICAL_PROD_TYPE, /* bundledProds */
                    null, /* addOptions */true);

            /*
             * Insert a bundle where the bundled products are prod1 and prod2. Prod2 has options so
             * we have to specify which configuration of the product needs to be added to the
             * bundle. (i.e. should we add the 4MB or the 8MB model). We specify this by setting the
             * encodedOptionValues attribute of the product.
             */
            // Choose the 8MB model. OptionId = 4. OptionValueId = 2.
            prod2.setEncodedOptionValues("4{2}");
            
            // Set the number of products we want in the bundle. 2 of prod1 and 1 of prod2.
            prod1.setBundledProdQuantity(2);
            prod2.setBundledProdQuantity(1);

            AdminProduct prod3 = insertProduct("bundle", KonakartAdminConstants.BUNDLE_PROD_TYPE, /* bundledProds */
                    new AdminProduct[]
                    { prod1, prod2 }, /* addOptions */false);

            if (prod3 != null)
            {
                /*
                 * Note that the name and description of the admin product object are set to null
                 * because they are in the AdminProductDescription array for each language. If the
                 * same product is read through the KonaKart application API, the name and
                 * description will be set to the values defined by the chosen language.
                 */
                System.out.println("Main Prod Id = " + prod3.getId());
                System.out.println("Main Prod Name = "
                        + prod3.getDescriptions()[0].getName());
                System.out.println("Main Prod Encoded Option Values  = "
                        + prod3.getEncodedOptionValues());


                // Get the products in the bundle
                AdminDataDescriptor dd = new AdminDataDescriptor();
                AdminProducts bundledProds = eng.getRelatedProducts(sessionId, dd, prod3.getId(),
                        KonakartAdminConstants.BUNDLED_PRODUCTS_RELATION,
                        KonakartAdminConstants.DEFAULT_LANGUAGE_ID);
                if (bundledProds != null && bundledProds.getProductArray() != null)
                {
                    for (int i = 0; i < bundledProds.getProductArray().length; i++)
                    {
                        AdminProduct prod = bundledProds.getProductArray()[i];
                        System.out.println("Bundled Prod Id = " + prod.getId());
                        System.out.println("Bundled Prod Name = "
                                + prod.getDescriptions()[0].getName());
                        System.out.println("Bundled Prod Encoded Option Values  = "
                                + prod.getEncodedOptionValues());
                    }
                }

            } else
            {
                System.out.println("The product could not be read from the DB");
            }

        } catch (Exception e)
        {
            e.printStackTrace();
        }

    }

    /**
     * Method called multiple times to insert a product
     * @param name
     * @param type
     * @param bundledProds
     * @param addOptions
     * @return Returns the inserted product after reading it back from the DB
     * @throws KKAdminException
     */
    static AdminProduct insertProduct(String name, int type, AdminProduct[] bundledProds,
            boolean addOptions) throws KKAdminException
    {

        /*
         * Instantiate an AdminProduct object and set some of its attributes
         */
        AdminProduct prod = new AdminProduct();

        // Set type
        prod.setType(type);
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

        /*
         * An Admin Product object has an array of AdminProductDescription objects, each one of
         * which contains a name, description and url for a different language. The url is language
         * dependent since it can point to the home page of the product for the country in question.
         */
        AdminProductDescription[] descriptions = new AdminProductDescription[2];
        descriptions[0] = new AdminProductDescription();
        descriptions[0].setDescription(name + " desc - E");
        descriptions[0].setLanguageId(1);
        descriptions[0].setName(name + " name - E");
        descriptions[0].setUrl("www.testprod.co.uk");
        descriptions[1] = new AdminProductDescription();
        descriptions[1].setDescription(name + " desc - D");
        descriptions[1].setLanguageId(2);
        descriptions[1].setName(name + " desc - D");
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

        if (addOptions)
        {
            /*
             * The following Admin Product Attributes are optional. They are used if the product has
             * options which can be selected to configure the product. These options may add or
             * subtract a value from the final price. i.e. Size "extra small" may be -$10.00 from
             * the product price while size "extra large" may be +$10.00.
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
            // This points to the option in the products_options table with id = 4 (Memory in demo
            // DB)
            prodAttr2.setOptionId(4);
            // This points to the option value in the products_options_values table with id = 2 (8MB
            // in demo DB)
            prodAttr2.setOptionValueId(2);
            // This option costs an extra 30 if selected
            prodAttr2.setPrice(new BigDecimal(30));
            prodAttr2.setPricePrefix("+");
            attrs[1] = prodAttr2;

            // Add the attributes to the product
            prod.setAttributes(attrs);
        }

        // Insert the product and get the product Id
        int prodId = eng.insertProduct(sessionId, prod);

        System.out.println("Product Id of inserted product = " + prodId);

        // Read the product from the database
        prod = eng.getProduct(sessionId, prodId);

        // Add the products to the bundle
        if (type == KonakartAdminConstants.BUNDLE_PROD_TYPE && bundledProds != null)
        {
            eng.addRelatedProducts(sessionId, bundledProds, prod.getId(),
                    KonakartAdminConstants.BUNDLED_PRODUCTS_RELATION);
        }
        return prod;
    }
}
