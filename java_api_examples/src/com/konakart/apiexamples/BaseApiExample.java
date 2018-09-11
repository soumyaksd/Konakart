//
// (c) 2006 DS Data Systems UK Ltd, All rights reserved.
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
package com.konakart.apiexamples;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.torque.TorqueException;

import com.konakart.app.EngineConfig;
import com.konakart.app.KKException;
import com.konakart.app.ProductSearch;
import com.konakart.appif.CountryIf;
import com.konakart.appif.KKEngIf;
import com.konakart.appif.ProductIf;
import com.konakart.appif.ProductSearchIf;
import com.konakart.appif.ProductsIf;
import com.konakart.bl.KKEngineUtils;
import com.konakart.db.KKBasePeer;
import com.konakart.util.KKCodeDescription;
import com.konakart.util.KKConstants;
import com.workingdogs.village.Record;

/**
 * Super class for the API Examples.
 */
public class BaseApiExample
{
    /** logger */
    public static Log log = LogFactory.getLog(BaseApiExample.class);

    /** The Application Engine */
    protected static KKEngIf eng;

    /**
     * When -1 is passed as a parameter for the language id, the engine uses the default language.
     */
    protected static final int DEFAULT_LANGUAGE = -1;

    /** POJO Engine */
    protected static final String KKEngName = "com.konakart.app.KKEng";

    /** SOAP Engine */
    protected static final String KKWSEngName = "com.konakart.app.KKWSEng";

    /** JAXWS Engine */
    protected static final String KKJAXWSEngName = "com.konakart.jws.KKJAXWSEng";

    /** RMI Engine */
    protected static final String KKRMIEngName = "com.konakart.rmi.KKRMIEng";

    /** JSON Engine */
    protected static final String KKJSONEngName = "com.konakart.json.KKJSONEng";

    /** POJO Custom Store Service Engine */
    protected static final String KKCustomStoreEngName = "com.konakart.bl.KKCustomStoreService";

    /** SOAP Custom Store Service Engine */
    protected static final String KKWSCustomStoreEngName = "com.konakart.ws.KKWSCustomStoreService";

    /** Default credentials for accessing the KonaKart Application Engine */
    private static String DEFAULT_USERNAME = "doe@konakart.com";

    private static String DEFAULT_PASSWORD = "password";

    /** StoreId - Used in Multi-Store Installations to identify the store */
    private static String storeId = KKConstants.KONAKART_DEFAULT_STORE_ID;

    /** Engine Mode - 0 = Single Store 1 = Multi-Store Multiple DBs, 2 = Multi-Store Single DB */
    private static int engineMode = KKConstants.MODE_SINGLE_STORE;

    /** DB Name of the Database we access - only used in MS-MDB mode */
    private static String dbName = null;

    /** debug flag */
    private static boolean debug = false;

    /** customersShared flag */
    private static boolean customersShared = false;

    /** productsShared flag */
    private static boolean productsShared = false;

    /** categoriesShared flag */
    private static boolean categoriesShared = false;

    /** The session id returned by a successful login */
    protected static String sessionId;

    /** konakart. configuration variables from konakart.properties **/
    private static Configuration konakartConfig = null;

    protected static final String DEF_STORE_ID = KKConstants.KONAKART_DEFAULT_STORE_ID;

    /** username */
    private static String username = DEFAULT_USERNAME;

    /** password */
    private static String password = DEFAULT_PASSWORD;
    
    protected static final String COMMON_USAGE = "                                 \n"
            + "   [-s storeId]       storeId (default " + DEF_STORE_ID + ")        \n"
            + "   [-m (0|1|2)]       engine Mode (default 0 = Single Store)        \n"
            + "   [-c (true|false)]  shared customers (default is false)           \n"
            + "   [-ps (true|false)] shared products (default is false)            \n"
            + "   [-cs (true|false)] shared categories (default is false)          \n"
            + "   [-d]               enable debug                                  \n";

    static protected void parseArgs(String[] args, String usage, int minArgs)
    {
        if (args == null || args.length < minArgs)
        {
            System.out.println("\n" + usage + "\nIllegal or insufficient arguments\n");
            System.exit(1);
            return;
        }

        for (int a = 0; a < args.length; a++)
        {
            if (args[a].equals("-m"))
            {
                setEngineMode(Integer.parseInt(args[++a]));
            } else if (args[a].equals("-s"))
            {
                setStoreId(args[++a]);
            } else if (args[a].equals("-c"))
            {
                setCustomersShared(args[++a].equalsIgnoreCase("true"));
            } else if (args[a].equals("-ps"))
            {
                setProductsShared(args[++a].equalsIgnoreCase("true"));
            } else if (args[a].equals("-cs"))
            {
                setCategoriesShared(args[++a].equalsIgnoreCase("true"));
            } else if (args[a].equals("-d"))
            {
                setDebug(true);
            } else
            {
                System.out.println("\n" + usage + "\nUnknown argument: " + args[a] + "\n");
                System.exit(1);
            }
        }

        if (isDebug())
        {
            System.out.println("store:             " + getStoreId());
            System.out.println("Engine Mode:       "
                    + KKCodeDescription.engineModeToString(getEngineMode()));
            System.out.println("Shared Customers?  " + isCustomersShared());
            System.out.println("Shared Products?   " + isProductsShared());
            System.out.println("Shared Categories? " + isCategoriesShared());
            System.out.println();
        }
    }

    /**
     * Initialise a KonaKart engine instance and perform a login to get a session id.
     * 
     * @throws ClassNotFoundException
     * @throws InstantiationException
     * @throws IllegalAccessException
     * @throws KKException
     * @throws ConfigurationException
     * @throws IOException
     * @throws InvocationTargetException
     * @throws IllegalArgumentException
     */
    static protected void init() throws ClassNotFoundException, InstantiationException,
            IllegalAccessException, KKException, ConfigurationException, IOException,
            IllegalArgumentException, InvocationTargetException
    {
        EngineConfig engConf = new EngineConfig();
        engConf.setMode(getEngineMode());
        engConf.setStoreId(getStoreId());
        engConf.setCustomersShared(isCustomersShared());
        engConf.setProductsShared(isProductsShared());
        engConf.setCategoriesShared(isCategoriesShared());

        /*
         * Instantiate a KonaKart Engine. Different engines can be instantiated by passing
         * KKWSEngName or KKRMIEngName or KKJSONEngName for the SOAP, RMI or JSON engines
         */
        eng = getKKEngByName(KKEngName, engConf);

        /*
         * Login with default credentials
         */
        sessionId = eng.login(getUsername(), getPassword());

        if (sessionId == null)
        {
            String msg = "Login of " + DEFAULT_USERNAME + " was unsuccessful";
            log.warn(msg);
            throw new KKException(msg);
        }

        if (getEngineMode() == EngineConfig.MODE_MULTI_STORE_NON_SHARED_DB)
        {
            dbName = getStoreId();
        }

        // Read the KonaKart config file - note that this is done in the engine but if we are
        // running this over SOAP we might not have access to the engine's Configuration data - so
        // we do it here explicitly on the client side.
        PropertiesConfiguration allConfig = new PropertiesConfiguration(
                KKConstants.KONAKART_PROPERTIES_FILE);
        if (allConfig.isEmpty())
        {
            throw new KKException("The configuration file: " + KKConstants.KONAKART_PROPERTIES_FILE
                    + " does not appear to contain any keys");
        }

        // Look for properties that are in the "konakart" namespace.
        Configuration subConf = allConfig.subset("konakart");
        if (subConf == null || subConf.isEmpty())
        {
            log.error("The konakart section in the properties file is missing. "
                    + "You must add at least one property to resolve this problem. "
                    + "e.g. konakart.session.expirationMinutes=30");
            return;
        }
        konakartConfig = subConf;
        log.info((new File(KKConstants.KONAKART_PROPERTIES_FILE)).getCanonicalPath() + " read");
    }

    protected static String getKonaKartConfig(String key)
    {
        String value = konakartConfig.getString(key);

        if (log.isInfoEnabled())
        {
            log.info(key + " = " + value);
        }
        return konakartConfig.getString(key);
    }

    /**
     * Look up the id of a country with the specified name
     * 
     * @param countryName
     * @return
     * @throws KKException
     */
    protected static int getCountryIdByName(String countryName) throws KKException
    {
        CountryIf country = eng.getCountryPerName(countryName);

        if (country == null)
        {
            throw new KKException("Country " + countryName + " not found in " + getStoreId());
        }

        return country.getId();
    }

    /**
     * Look up the id of a product with the specified name
     * 
     * @param name
     * @return
     * @throws KKException
     */
    protected static int getProductIdByName(String name) throws KKException
    {
        ProductSearchIf pSearch = new ProductSearch();
        pSearch.setSearchText(name);

        ProductsIf prods = eng.searchForProducts(sessionId, null, pSearch, -1);
        ProductIf[] prod = prods.getProductArray();

        for (int p = 0; p < prod.length; p++)
        {
            // This just picks the first product that starts with our search name... be careful
            if (prod[p].getName().toUpperCase().startsWith(name.toUpperCase()))
            {
                System.out.println("Product with name '" + name + "' found - id = "
                        + prod[p].getId());
                return prod[p].getId();
            }
        }

        throw new KKException("Could not find a product with name = " + name);
    }

    /**
     * Execute statement that works for multi database testing
     * 
     * @param sqlStatement
     * @throws TorqueException
     */
    protected static void executeStatement(String sqlStatement) throws TorqueException
    {
        String statement = adjustQueryForMode(sqlStatement);

        if (dbName == null || getEngineMode() != EngineConfig.MODE_MULTI_STORE_NON_SHARED_DB)
        {
            KKBasePeer.executeStatement(statement);
        } else
        {
            KKBasePeer.executeStatement(statement, dbName);
        }
    }

    /**
     * The store_id may be added to the query depending on the mode
     * 
     * @param query
     * @return
     */
    protected static String adjustQueryForMode(String query)
    {
        String statement = query;

        if (getEngineMode() == EngineConfig.MODE_MULTI_STORE_SHARED_DB)
        {
            if (statement.contains("where") || statement.contains("WHERE")
                    || statement.contains("Where"))
            {
                statement = statement + " and store_id = '" + getStoreId() + "'";
            } else
            {
                statement = statement + " where store_id = '" + getStoreId() + "'";
            }
        }

        return statement;
    }

    /**
     * Execute query that works for multi database testing
     * 
     * @param sqlQuery
     * @throws TorqueException
     */
    protected static List<Record> executeQuery(String sqlQuery) throws TorqueException
    {
        String query = adjustQueryForMode(sqlQuery);

        if (dbName == null || getEngineMode() != EngineConfig.MODE_MULTI_STORE_NON_SHARED_DB)
        {
            return KKBasePeer.executeQuery(query);
        }

        return KKBasePeer.executeQuery(query, dbName);
    }

    /**
     * Utility method to instantiate an engine instance. The class name of the engine is passed in
     * as a parameter so this method may be used to instantiate a POJO engine, a SOAP engine, an RMI
     * engine or a JSON engine.
     * 
     * @param engineClassName
     * @param config
     * @return Returns an Engine Instance
     * @throws IllegalArgumentException
     * @throws InstantiationException
     * @throws IllegalAccessException
     * @throws InvocationTargetException
     * @throws ClassNotFoundException
     */
    protected static KKEngIf getKKEngByName(String engineClassName, EngineConfig config)
            throws IllegalArgumentException, InstantiationException, IllegalAccessException,
            InvocationTargetException, ClassNotFoundException
    {
        return new KKEngineUtils().getKKEngByName(engineClassName, config);
    }

    /**
     * @return the engineMode
     */
    protected static int getEngineMode()
    {
        return engineMode;
    }

    /**
     * @param engineMode
     *            the engineMode to set
     */
    protected static void setEngineMode(int engineMode)
    {
        BaseApiExample.engineMode = engineMode;
    }

    /**
     * @return the sessionId
     */
    protected static String getSessionId()
    {
        return sessionId;
    }

    /**
     * @param sessionId
     *            the sessionId to set
     */
    protected static void setSessionId(String sessionId)
    {
        BaseApiExample.sessionId = sessionId;
    }

    /**
     * @return the storeId
     */
    protected static String getStoreId()
    {
        return storeId;
    }

    /**
     * @param storeId
     *            the storeId to set
     */
    protected static void setStoreId(String storeId)
    {
        BaseApiExample.storeId = storeId;
    }

    /**
     * @return the debug
     */
    protected static boolean isDebug()
    {
        return debug;
    }

    /**
     * @param debug
     *            the debug to set
     */
    protected static void setDebug(boolean debug)
    {
        BaseApiExample.debug = debug;
    }

    /**
     * @return the customersShared
     */
    protected static boolean isCustomersShared()
    {
        return customersShared;
    }

    /**
     * @param customersShared
     *            the customersShared to set
     */
    protected static void setCustomersShared(boolean customersShared)
    {
        BaseApiExample.customersShared = customersShared;
    }

    /**
     * @return the dEFAULT_USERNAME
     */
    protected static String getDEFAULT_USERNAME()
    {
        return DEFAULT_USERNAME;
    }

    /**
     * @return the productsShared
     */
    public static boolean isProductsShared()
    {
        return productsShared;
    }

    /**
     * @param productsShared
     *            the productsShared to set
     */
    public static void setProductsShared(boolean productsShared)
    {
        BaseApiExample.productsShared = productsShared;
    }

    /**
     * @return the eng
     */
    public static KKEngIf getEng()
    {
        return eng;
    }

    /**
     * @return the categoriesShared
     */
    public static boolean isCategoriesShared()
    {
        return categoriesShared;
    }

    /**
     * @param categoriesShared
     *            the categoriesShared to set
     */
    public static void setCategoriesShared(boolean categoriesShared)
    {
        BaseApiExample.categoriesShared = categoriesShared;
    }

    /**
     * @return the username
     */
    public static String getUsername()
    {
        return username;
    }

    /**
     * @param username the username to set
     */
    public static void setUsername(String username)
    {
        BaseApiExample.username = username;
    }

    /**
     * @return the password
     */
    public static String getPassword()
    {
        return password;
    }

    /**
     * @param password the password to set
     */
    public static void setPassword(String password)
    {
        BaseApiExample.password = password;
    }
}