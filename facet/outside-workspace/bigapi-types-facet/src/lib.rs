use chrono::{NaiveDate, NaiveDateTime};
use facet::Facet;
use uuid::Uuid;

/// The root struct representing the catalog of everything.
#[derive(Facet, Clone)]
pub struct Catalog {
    pub id: Uuid,
    pub businesses: Vec<Business>,
    pub created_at: NaiveDateTime,
    pub metadata: CatalogMetadata,
}

#[derive(Facet, Clone)]
pub struct CatalogMetadata {
    pub version: String,
    pub region: String,
}

/// A business represented in the catalog.
#[derive(Facet, Clone)]
pub struct Business {
    pub id: Uuid,
    pub name: String,
    pub address: Address,
    pub owner: BusinessOwner,
    pub users: Vec<BusinessUser>,
    pub branches: Vec<Branch>,
    pub products: Vec<Product>,
    pub created_at: NaiveDateTime,
}

#[derive(Facet, Clone)]
pub struct BusinessOwner {
    pub user: User,
    pub ownership_percent: f32,
}

#[derive(Facet, Clone)]
pub struct Branch {
    pub id: Uuid,
    pub name: String,
    pub address: Address,
    pub employees: Vec<BusinessUser>,
    pub inventory: Vec<BranchInventory>,
    pub open: bool,
}

#[derive(Facet, Clone)]
pub struct BranchInventory {
    pub product: Product,
    pub stock: u32,
    pub location_code: Option<String>,
}

/// A user of the business
#[derive(Facet, Clone)]
pub struct BusinessUser {
    pub user: User,
    pub roles: Vec<Role>,
    pub is_active: bool,
    pub created_at: NaiveDateTime,
}

#[derive(Facet, Clone)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub profile: UserProfile,
    pub settings: Settings,
}

#[derive(Facet, Clone)]
pub struct UserProfile {
    pub first_name: String,
    pub last_name: String,
    pub date_of_birth: NaiveDate,
    pub gender: Gender,
    pub bio: Option<String>,
    pub avatar_url: Option<String>,
    pub home_address: Address,
}

#[derive(Facet, Clone)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub state: String,
    pub postal_code: String,
    pub country: String,
    pub geo: Option<GeoLocation>,
}

#[derive(Facet, Clone)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Facet, Clone)]
#[repr(u8)]
pub enum Gender {
    Male,
    Female,
    Other,
    PreferNotToSay,
}

#[derive(Facet, Clone)]
pub struct Product {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub price_cents: u64,
    pub currency: String,
    pub available: bool,
    pub metadata: Option<ProductMetadata>,
    pub reviews: Vec<ProductReview>,
    pub categories: Vec<Category>,
}

#[derive(Facet, Clone)]
pub struct ProductMetadata {
    pub sku: Option<String>,
    pub categories: Vec<String>,
    pub weight_grams: Option<u32>,
    pub dimensions: Option<ProductDimensions>,
}

#[derive(Facet, Clone)]
pub struct ProductDimensions {
    pub length_mm: Option<f32>,
    pub width_mm: Option<f32>,
    pub height_mm: Option<f32>,
}

#[derive(Facet, Clone)]
pub struct ProductReview {
    pub id: Uuid,
    pub reviewer: UserSummary,
    pub rating: u8,
    pub text: Option<String>,
    pub created_at: NaiveDateTime,
}

#[derive(Facet, Clone)]
pub struct Category {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub parent: Option<Box<Category>>,
}

/// Brief user reference (for lists)
#[derive(Facet, Clone)]
pub struct UserSummary {
    pub id: Uuid,
    pub username: String,
    pub avatar_url: Option<String>,
}

#[derive(Facet, Clone)]
pub struct Role {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub permissions: Vec<Permission>,
}

#[derive(Facet, Clone)]
pub struct Permission {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
}

#[derive(Facet, Clone)]
pub struct Settings {
    pub user_id: Uuid,
    pub email_notifications: bool,
    pub push_notifications: bool,
    pub theme: Theme,
    pub language: String,
}

#[derive(Facet, Clone)]
#[repr(u8)]
pub enum Theme {
    Light,
    Dark,
    System,
}

// -----------------------------------
// mock data generator
// -----------------------------------

pub fn generate_mock_catalog() -> Catalog {
    use chrono::{NaiveDate, NaiveDateTime};
    use uuid::Uuid;

    // Helper for now
    fn now() -> NaiveDateTime {
        chrono::Utc::now().naive_utc()
    }
    fn today() -> NaiveDate {
        chrono::Utc::now().date_naive()
    }
    fn mock_address() -> Address {
        Address {
            street: "123 Main St.".to_string(),
            city: "Metropolis".to_string(),
            state: "Stateville".to_string(),
            postal_code: "12345".to_string(),
            country: "Countryland".to_string(),
            geo: Some(GeoLocation {
                latitude: 51.0,
                longitude: -0.1,
            }),
        }
    }
    fn mock_role() -> Role {
        Role {
            id: Uuid::new_v4(),
            name: "Employee".to_string(),
            description: Some("Regular employee role".to_string()),
            permissions: vec![Permission {
                id: Uuid::new_v4(),
                name: "access_dashboard".to_string(),
                description: Some("Can access the dashboard".to_string()),
            }],
        }
    }
    fn mock_user(idx: u32) -> User {
        User {
            id: Uuid::new_v4(),
            username: format!("user{}", idx),
            email: format!("user{}@email.com", idx),
            created_at: now(),
            updated_at: now(),
            profile: UserProfile {
                first_name: format!("First{}", idx),
                last_name: format!("Last{}", idx),
                date_of_birth: today(),
                gender: if idx % 2 == 0 {
                    Gender::Male
                } else {
                    Gender::Female
                },
                bio: Some(format!("Bio of user {}", idx)),
                avatar_url: None,
                home_address: mock_address(),
            },
            settings: Settings {
                user_id: Uuid::new_v4(),
                email_notifications: true,
                push_notifications: idx % 2 == 0,
                theme: if idx % 2 == 0 {
                    Theme::Light
                } else {
                    Theme::Dark
                },
                language: "en-US".to_string(),
            },
        }
    }
    fn mock_user_summary(idx: u32) -> UserSummary {
        UserSummary {
            id: Uuid::new_v4(),
            username: format!("user{}", idx),
            avatar_url: None,
        }
    }
    fn mock_category(id: u8) -> Category {
        if id == 0 {
            Category {
                id: Uuid::new_v4(),
                name: "Root Category".to_string(),
                description: Some("Top of the tree".to_string()),
                parent: None,
            }
        } else {
            Category {
                id: Uuid::new_v4(),
                name: format!("Subcategory {}", id),
                description: Some(format!("Subcategory number {}", id)),
                parent: Some(Box::new(mock_category(id - 1))),
            }
        }
    }
    fn mock_product(idx: u32) -> Product {
        Product {
            id: Uuid::new_v4(),
            name: format!("Product{}", idx),
            description: Some(format!("Description for product {}", idx)),
            price_cents: (1000 + (idx * 100)) as u64,
            currency: "USD".to_string(),
            available: true,
            metadata: Some(ProductMetadata {
                sku: Some(format!("SKU{}", idx)),
                categories: vec!["Electronics".to_string(), "Gadgets".to_string()],
                weight_grams: Some(500 + 10 * idx),
                dimensions: Some(ProductDimensions {
                    length_mm: Some(100.0 + idx as f32),
                    width_mm: Some(50.0 + idx as f32),
                    height_mm: Some(25.5 + idx as f32),
                }),
            }),
            reviews: vec![ProductReview {
                id: Uuid::new_v4(),
                reviewer: mock_user_summary(idx),
                rating: 4 + ((idx % 2) as u8),
                text: Some(format!("Review for product {}", idx)),
                created_at: now(),
            }],
            categories: vec![mock_category(idx as u8)],
        }
    }

    // Construct mock business users
    let business_users: Vec<BusinessUser> = (1..=2)
        .map(|i| BusinessUser {
            user: mock_user(i),
            roles: vec![mock_role()],
            is_active: true,
            created_at: now(),
        })
        .collect();

    // mock owner
    let owner = BusinessOwner {
        user: mock_user(100),
        ownership_percent: 100.0,
    };

    // mock branch with inventory
    let products: Vec<Product> = (1..=3).map(mock_product).collect();
    let branch_inventory: Vec<BranchInventory> = products
        .iter()
        .cloned()
        .map(|p| BranchInventory {
            product: p,
            stock: 50,
            location_code: Some("A-01".to_string()),
        })
        .collect();

    let branch = Branch {
        id: Uuid::new_v4(),
        name: "Central Branch".to_string(),
        address: mock_address(),
        employees: business_users.clone(),
        inventory: branch_inventory,
        open: true,
    };

    let business = Business {
        id: Uuid::new_v4(),
        name: "Awesome Business".to_string(),
        address: mock_address(),
        owner,
        users: business_users,
        branches: vec![branch],
        products,
        created_at: now(),
    };

    Catalog {
        id: Uuid::new_v4(),
        businesses: vec![business],
        created_at: now(),
        metadata: CatalogMetadata {
            version: "1.0.1!".to_string(),
            region: "US".to_string(),
        },
    }
}
